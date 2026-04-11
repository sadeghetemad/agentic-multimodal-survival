import boto3
import pandas as pd
from typing import Optional, Dict
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup


class SimpleCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


class PatientFeatureService:

    def __init__(
        self,
        region: str,
        genomic_fg_name: str,
        clinical_fg_name: str,
        imaging_fg_name: str,
        bucket: str,
        prefix: str,
        use_online_store: bool = False
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
        self.cache = SimpleCache()

    # MAIN ENTRY
    def get_patient_features(self, patient_id: str) -> Optional[pd.DataFrame]:

        # Cache check
        cached = self.cache.get(patient_id)
        if cached is not None:
            print(f"[FeatureService] Cache hit for {patient_id}")
            return cached

        # Route based on mode
        if self.use_online_store:
            df = self._get_from_online_store(patient_id)
        else:
            df = self._get_from_athena(patient_id)

        if df is None or df.empty:
            return None

        # Clean & standardize
        df = self._clean_columns(df)

        # ensure single row
        df = df.iloc[[0]]

        # Cache result
        self.cache.set(patient_id, df)

        return df

    # ATHENA Full Join Query
    def _get_from_athena(self, patient_id: str) -> Optional[pd.DataFrame]:

        # basic sanitization
        patient_id = patient_id.replace("'", "")

        query_string = f"""
            SELECT g.*, c.*, i.*
            FROM "{self.genomic_table}" g
            LEFT JOIN "{self.clinical_table}" c
                ON g.case_id = c.case_id
            LEFT JOIN "{self.imaging_table}" i
                ON c.case_id = i.subject
            WHERE g.case_id = '{patient_id}'
            """

        print(f"[FeatureService] Athena query for {patient_id}")

        self.genomic_query.run(
            query_string=query_string,
            output_location=self.output_location
        )
        self.genomic_query.wait()

        df = self.genomic_query.as_dataframe()

        return df

    # ONLINE STORE
    def _get_from_online_store(self, patient_id: str) -> Optional[pd.DataFrame]:

        print(f"[FeatureService] Online store fetch for {patient_id}")

        try:
            genomic = self._get_record(self.genomic_fg.name, patient_id, "case_id")
            clinical = self._get_record(self.clinical_fg.name, patient_id, "case_id")
            imaging = self._get_record(self.imaging_fg.name, patient_id, "subject")

            if not genomic:
                return None

            merged = {**genomic, **clinical, **imaging}

            return pd.DataFrame([merged])

        except Exception as e:
            print(f"[FeatureService] Online fetch error: {e}")
            return None

    def _get_record(self, fg_name: str, record_id: str, key: str) -> Dict:

        response = self.featurestore_runtime.get_record(
            FeatureGroupName=fg_name,
            RecordIdentifierValueAsString=record_id
        )

        record = response.get("Record", [])

        result = {}

        for item in record:
            name = item["FeatureName"]
            value = item.get("ValueAsString", None)

            if value is not None:
                try:
                    value = float(value)
                except:
                    pass

            result[name] = value

        return result

    # CLEANING
    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        drop_patterns = [
            'eventtime',
            'write_time',
            'api_invocation_time',
            'is_deleted'
        ]

        cols_to_drop = []

        for col in df.columns:

            if any(p in col for p in drop_patterns):
                cols_to_drop.append(col)

            if col.startswith('case_id'):
                cols_to_drop.append(col)

            if 'diagnostics' in col:
                cols_to_drop.append(col)

        cols_to_drop += ['imagename', 'maskname', 'subject']

        df = df.drop(columns=list(set(cols_to_drop)), errors='ignore')

        return df