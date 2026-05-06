import json
import os
import sys
from typing import Any, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.feature_service import PatientFeatureService
from config.settings import (
    AWS_REGION,
    GENOMIC_FG,
    CLINICAL_FG,
    IMAGING_FG,
    BUCKET,
    PREFIX,
)


_feature_service = None


def get_feature_service() -> PatientFeatureService:
    global _feature_service

    if _feature_service is None:
        _feature_service = PatientFeatureService(
            region=AWS_REGION,
            genomic_fg_name=GENOMIC_FG,
            clinical_fg_name=CLINICAL_FG,
            imaging_fg_name=IMAGING_FG,
            bucket=BUCKET,
            prefix=PREFIX,
            use_online_store=False
        )

    return _feature_service


def _extract_arguments(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {}

    if isinstance(event.get("arguments"), dict):
        return event["arguments"]

    params = event.get("params")
    if isinstance(params, dict) and isinstance(params.get("arguments"), dict):
        return params["arguments"]

    # fallback for direct invocation
    return event


def lambda_handler(event, context):
    print("FETCH_PATIENT EVENT:", json.dumps(event, default=str))

    try:
        args = _extract_arguments(event)
        patient_id = args.get("patient_id")

        if not patient_id:
            return {
                "status": "error",
                "message": "patient_id is required"
            }

        feature_service = get_feature_service()
        result = feature_service.get_patient_features(str(patient_id))

        # result shape from your service:
        # {
        #   "status": "ok",
        #   "data": {"features": {...}}
        # }
        if not isinstance(result, dict):
            return {
                "status": "error",
                "message": "Invalid response from feature service"
            }

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"fetch_patient_lambda failed: {str(e)}"
        }