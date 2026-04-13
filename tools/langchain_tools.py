from langchain_core.tools import tool
from typing import Dict

from services.feature_parser_service import parse
from services.feature_completion_service import complete
from services.feature_validator_service import validate
from services.feature_service import PatientFeatureService
from services.prediction_pipeline import predict_multimodal
from config.settings import *


# -------------------------
# INIT FEATURE SERVICE
# -------------------------
feature_service = PatientFeatureService(
    region=AWS_REGION,
    genomic_fg_name=GENOMIC_FG,
    clinical_fg_name=CLINICAL_FG,
    imaging_fg_name=IMAGING_FG,
    bucket=BUCKET,
    prefix=PREFIX
)


# -------------------------
# PARSE
# -------------------------
@tool
def parse_features(text: str) -> Dict:
    """
    Extract structured medical features from raw clinical text.

    INPUT:
        text (string)

    OUTPUT:
        {
            "status": "ok",
            "data": {
                "features": {...}
            }
        }
    """

    if not text:
        return {
            "status": "error",
            "message": "Empty input text"
        }

    try:
        features = parse(text)

        if not features or len(features) == 0:
            return {
                "status": "error",
                "message": "❌ No medical information found in the input. Please provide relevant clinical data such as age, smoking status, tumor size, or other medical features."
            }

        return {
            "status": "ok",
            "data": {
                "features": features
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Parse failed: {str(e)}"
        }


# -------------------------
# VALIDATE
# -------------------------
@tool
def validate_features(features: Dict) -> Dict:
    """
    Validate extracted features.

    INPUT:
        features (dict)

    OUTPUT:
        {
            "status": "ok",
            "data": {
                "features": {...}
            }
        }
    """

    if not isinstance(features, dict) or not features:
        return {
            "status": "error",
            "message": "Invalid or empty features"
        }

    return validate(features)


# -------------------------
# COMPLETE
# -------------------------
@tool
def complete_features(features: Dict) -> Dict:
    """
    Fill missing features using similarity-based completion.

    INPUT:
        features (dict)

    OUTPUT:
        {
            "status": "ok",
            "data": {
                "features": {...}
            }
        }
    """

    if not isinstance(features, dict) or not features:
        return {
            "status": "error",
            "message": "Invalid or empty features"
        }

    try:
        completed = complete(features)

        if isinstance(completed, dict) and completed.get("status") == "error":
            return completed

        return {
            "status": "ok",
            "data": {
                "features": completed
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Completion failed: {str(e)}"
        }


# -------------------------
# FETCH PATIENT
# -------------------------
@tool
def fetch_patient(patient_id: str) -> Dict:
    """
    Fetch patient features
    """
    if not patient_id:
        return {
            "status": "error",
            "message": "Missing patient_id"
        }

    result = feature_service.get_patient_features(patient_id)

    if "features" in result:
        features = result.get("features", {})
    else:
        features = result.get("data", {}).get("features", {})

    return {
        "status": "ok",
        "features": features
    }

# -------------------------
# PREDICT
# -------------------------
@tool
def predict(features: Dict) -> Dict:
    """
    Predict NSCLC survival risk using full multimodal pipeline.
    """

    if not isinstance(features, dict) or not features:
        return {
            "status": "error",
            "message": "Invalid features"
        }

    result = predict_multimodal(features)

    return result