import time
import threading
from typing import Optional, Dict, Any

import boto3
import pandas as pd
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup


class SimpleCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.store: Dict[str, Any] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            item = self.store.get(key)
            if item is None:
                return None

            value, expires_at = item

            if time.time() > expires_at:
                del self.store[key]
                return None

            return value

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            if len(self.store) >= self.max_size:
                oldest_key = next(iter(self.store))
                del self.store[oldest_key]

            expires_at = time.time() + self.ttl_seconds
            self.store[key] = (value, expires_at)

    def clear(self) -> None:
        with self.lock:
            self.store.clear()


class PatientFeatureService:
    def __init__(
        self,
        region: str,
        genomic_fg_name: str,
        clinical_fg_name: str,
        imaging_fg_name: str,
        bucket: str,
        prefix: str,
        use_online_store: bool = False,
        cache_ttl_seconds: int = 300,
        cache_max_size: int = 1000
    ):
        self.region = region
        self.use_online_store = use_online_store

        # AWS Session
        boto_session = boto3.Session(region_name=region)

        self.sagemaker_client = boto_session.client("sagemaker")
        self.featurestore_runtime = boto_session.client(
            "sagemaker-featurestore-runtime"
        )

        self.feature_store_session = Session(
            boto_session=boto_session,
            sagemaker_client=self.sagemaker_client,
            sagemaker_featurestore_runtime_client=self.featurestore_runtime
        )

        # Feature Groups
        self.genomic_fg = FeatureGroup(
            name=genomic_fg_name,
            sagemaker_session=self.feature_store_session
        )
        self.clinical_fg = FeatureGroup(
            name=clinical_fg_name,
            sagemaker_session=self.feature_store_session
        )
        self.imaging_fg = FeatureGroup(
            name=imaging_fg_name,
            sagemaker_session=self.feature_store_session
        )

        # Athena Query Setup
        self.genomic_query = self.genomic_fg.athena_query()
        self.clinical_query = self.clinical_fg.athena_query()
        self.imaging_query = self.imaging_fg.athena_query()

        self.genomic_table = self.genomic_query.table_name
        self.clinical_table = self.clinical_query.table_name
        self.imaging_table = self.imaging_query.table_name

        self.output_location = f"s3://{bucket}/{prefix}/feature-store-queries"

        # Cache
        self.cache = SimpleCache(
            ttl_seconds=cache_ttl_seconds,
            max_size=cache_max_size
        )

    def get_patient_features(self, patient_id: str) -> Dict[str, Any]:
        patient_id = self._sanitize_patient_id(patient_id)

        if not patient_id:
            return {
                "status": "error",
                "message": "patient_id is required"
            }

        cache_key = self._build_cache_key(patient_id)

        cached = self.cache.get(cache_key)
        if cached is not None:
            print(f"[FeatureService] Cache hit for {patient_id}")
            return cached

        if self.use_online_store:
            df = self._get_from_online_store(patient_id)
        else:
            df = self._get_from_athena(patient_id)

        if df is None or df.empty:
            result = {
                "status": "error",
                "message": f"No data found for patient {patient_id}"
            }
            self.cache.set(cache_key, result)
            return result

        df = self._clean_columns(df)

        if df.empty:
            result = {
                "status": "error",
                "message": f"No usable data found for patient {patient_id}"
            }
            self.cache.set(cache_key, result)
            return result

        df = df.iloc[[0]]
        record = df.to_dict(orient="records")[0]

        result = {
            "status": "ok",
            "data": {
                "features": record
            }
        }

        self.cache.set(cache_key, result)
        return result

    def _get_from_athena(self, patient_id: str) -> Optional[pd.DataFrame]:
        print(f"[FeatureService] Athena query for {patient_id}")

        query_string = f"""
            SELECT g.*, c.*, i.*
            FROM "{self.genomic_table}" g
            LEFT JOIN "{self.clinical_table}" c
                ON g.case_id = c.case_id
            LEFT JOIN "{self.imaging_table}" i
                ON c.case_id = i.subject
            WHERE g.case_id = '{patient_id}'
        """

        try:
            self.genomic_query.run(
                query_string=query_string,
                output_location=self.output_location
            )
            self.genomic_query.wait()
            df = self.genomic_query.as_dataframe()
            return df

        except Exception as e:
            print(f"[FeatureService] Athena fetch error: {e}")
            return None

    def _get_from_online_store(self, patient_id: str) -> Optional[pd.DataFrame]:
        print(f"[FeatureService] Online store fetch for {patient_id}")

        try:
            genomic = self._get_record(
                fg_name=self.genomic_fg.name,
                record_id=patient_id,
                expected_key="case_id"
            )
            clinical = self._get_record(
                fg_name=self.clinical_fg.name,
                record_id=patient_id,
                expected_key="case_id"
            )
            imaging = self._get_record(
                fg_name=self.imaging_fg.name,
                record_id=patient_id,
                expected_key="subject"
            )

            if not genomic:
                return None

            merged = {**genomic, **clinical, **imaging}
            return pd.DataFrame([merged])

        except Exception as e:
            print(f"[FeatureService] Online fetch error: {e}")
            return None

    def _get_record(
        self,
        fg_name: str,
        record_id: str,
        expected_key: str
    ) -> Dict[str, Any]:
        response = self.featurestore_runtime.get_record(
            FeatureGroupName=fg_name,
            RecordIdentifierValueAsString=record_id
        )

        record = response.get("Record", [])
        result: Dict[str, Any] = {}

        for item in record:
            name = item["FeatureName"]
            value = item.get("ValueAsString")

            if value is not None:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass

            result[name] = value

        # اگر کلید اصلی اصلاً در رکورد نبود، این رکورد مشکوک است
        if expected_key not in result:
            print(
                f"[FeatureService] Warning: expected key '{expected_key}' "
                f"not found in feature group '{fg_name}' for record '{record_id}'"
            )

        return result

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_patterns = [
            "eventtime",
            "write_time",
            "api_invocation_time",
            "is_deleted"
        ]

        cols_to_drop = []

        for col in df.columns:
            col_lower = col.lower()

            if any(pattern in col_lower for pattern in drop_patterns):
                cols_to_drop.append(col)

            if col_lower.startswith("case_id"):
                cols_to_drop.append(col)

            if "diagnostics" in col_lower:
                cols_to_drop.append(col)

        cols_to_drop += ["imagename", "maskname", "subject"]

        df = df.drop(columns=list(set(cols_to_drop)), errors="ignore")

        # حذف ستون‌های تکراری احتمالی
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def _sanitize_patient_id(self, patient_id: str) -> str:
        if patient_id is None:
            return ""

        return str(patient_id).strip().replace("'", "")

    def _build_cache_key(self, patient_id: str) -> str:
        return (
            f"{self.region}|"
            f"{self.use_online_store}|"
            f"{self.genomic_fg.name}|"
            f"{self.clinical_fg.name}|"
            f"{self.imaging_fg.name}|"
            f"{patient_id}"
        )