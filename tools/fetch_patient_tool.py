from services.feature_service import PatientFeatureService
from config.settings import *

feature_service = PatientFeatureService(
    region=AWS_REGION,
    genomic_fg_name=GENOMIC_FG,
    clinical_fg_name=CLINICAL_FG,
    imaging_fg_name=IMAGING_FG,
    bucket=BUCKET,
    prefix=PREFIX
)


def get_patient_features(patient_id: str):
    df = feature_service.get_patient_features(patient_id)

    # -------------------------
    # Validation
    # -------------------------
    if df is None or df.empty:
        return {
            "status": "error",
            "message": "Patient not found"
        }

    if len(df) > 1:
        df = df.iloc[[0]]  # just take first safely

    # -------------------------
    # Clean
    # -------------------------
    record = df.iloc[0].fillna(0).to_dict()

    # optional: remove unwanted fields
    blacklist = ["eventtime", "write_time", "api_invocation_time"]
    record = {k: v for k, v in record.items() if k not in blacklist}

    # -------------------------
    # Structured output
    # -------------------------
    return {
        "status": "ok",
        "patient_id": patient_id,
        "features": record
    }