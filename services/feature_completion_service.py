import numpy as np
import boto3
import joblib
import os

from services.feature_service import PatientFeatureService
from config.settings import *


# Init Feature Service
feature_service = PatientFeatureService(
    region=AWS_REGION,
    genomic_fg_name=GENOMIC_FG,
    clinical_fg_name=CLINICAL_FG,
    imaging_fg_name=IMAGING_FG,
    bucket=BUCKET,
    prefix=PREFIX
)


# Load Feature order
def load_feature_order():

    filename = "feature_order.joblib"

    if not os.path.exists(filename):
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/{filename}",
            filename
        )

    return joblib.load(filename)


feature_order = load_feature_order()


# Cache
FEATURE_MATRIX = None

def load_all_patients():

    global FEATURE_MATRIX

    if FEATURE_MATRIX is not None:
        return FEATURE_MATRIX

    query = f"""
            SELECT *
            FROM "{feature_service.genomic_table}" g
            LEFT JOIN "{feature_service.clinical_table}" c
                ON g.case_id = c.case_id
            LEFT JOIN "{feature_service.imaging_table}" i
                ON c.case_id = i.subject
        """

    feature_service.genomic_query.run(
        query_string=query,
        output_location=feature_service.output_location
    )
    feature_service.genomic_query.wait()

    df = feature_service.genomic_query.as_dataframe()

    df = feature_service._clean_columns(df)

    df = df.reindex(columns=feature_order, fill_value=0)

    FEATURE_MATRIX = df.values.astype("float32")

    print(f"[FeatureCompletionService] Loaded {len(FEATURE_MATRIX)} patients")

    return FEATURE_MATRIX


# Core Logic
def complete(features: dict):

    if not features:
        raise ValueError("Empty features")

    matrix = load_all_patients()

    # Create feature vector
    vector = np.array([float(features.get(f, 0)) for f in feature_order])

    # Mask for Known features
    mask = np.array([1 if f in features else 0 for f in feature_order])

    # Similarity
    sims = np.dot(matrix * mask, vector * mask) / (
        np.linalg.norm(matrix * mask, axis=1) *
        np.linalg.norm(vector * mask) + 1e-8
    )

    idx = int(np.argmax(sims))
    nearest = matrix[idx]

    # fill
    completed = {}

    for i, f in enumerate(feature_order):
        if f in features:
            completed[f] = float(features[f])
        else:
            completed[f] = float(nearest[i])

    return completed