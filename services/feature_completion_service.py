import numpy as np
import boto3
import joblib
import os

from services.feature_service import PatientFeatureService
from config.settings import *


CACHE_PATH = "feature_matrix.pkl"

_feature_service = None

def get_feature_service():
    global _feature_service
    if _feature_service is None:
        _feature_service = PatientFeatureService(
            region=AWS_REGION,
            genomic_fg_name=GENOMIC_FG,
            clinical_fg_name=CLINICAL_FG,
            imaging_fg_name=IMAGING_FG,
            bucket=BUCKET,
            prefix=PREFIX
        )
    return _feature_service


MIN_FEATURES = 3
TOP_K = 5
SIM_THRESHOLD = 0.5

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


feature_order = None

def get_feature_order():
    global feature_order
    if feature_order is None:
        feature_order = load_feature_order()
    return feature_order


FEATURE_MATRIX = None

def load_all_patients():

    global FEATURE_MATRIX

    # -------------------------
    # MEMORY CACHE
    # -------------------------
    if FEATURE_MATRIX is not None:
        return FEATURE_MATRIX

    # -------------------------
    # DISK CACHE
    # -------------------------
    if os.path.exists(CACHE_PATH):
        FEATURE_MATRIX = joblib.load(CACHE_PATH)
        print("[CACHE] Loaded from disk")
        return FEATURE_MATRIX

    # -------------------------
    # DB QUERY
    # -------------------------
    print("[DB] Loading from Athena...")

    feature_service = get_feature_service()

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
    feature_order = get_feature_order()
    df = df.reindex(columns=feature_order, fill_value=0)

    FEATURE_MATRIX = df.values.astype("float32")

    # save to disk
    joblib.dump(FEATURE_MATRIX, CACHE_PATH)

    print(f"[DB] Loaded {len(FEATURE_MATRIX)} patients")

    return FEATURE_MATRIX

# Core Logic
def complete(features: dict):

    if not features:
        raise ValueError("Empty features")

    matrix = load_all_patients()

    print("👉 Input features:", features)

    # -------------------------
    # VECTOR + MASK
    # -------------------------
    feature_order = get_feature_order()
    vector = np.array(
        [float(features.get(f, 0.0)) for f in feature_order],
        dtype=float
    )

    mask = np.array(
        [1.0 if f in features else 0.0 for f in feature_order],
        dtype=float
    )

    num_known = int(mask.sum())
    print(f"👉 Known features: {num_known}")

    # -------------------------
    # GUARD 1: MIN FEATURES
    # -------------------------
    if num_known < MIN_FEATURES:
        return {
            "status": "error",
            "message": f"❌ Not enough medical data ({num_known}). Need at least {MIN_FEATURES} features."
        }

    # -------------------------
    # MASKED SIMILARITY
    # -------------------------
    masked_matrix = matrix * mask
    masked_vector = vector * mask

    denom = (
        np.linalg.norm(masked_matrix, axis=1) *
        np.linalg.norm(masked_vector) + 1e-8
    )

    sims = np.dot(masked_matrix, masked_vector) / denom

    max_sim = float(np.max(sims))
    mean_sim = float(np.mean(sims))

    print(f"👉 Similarity max={max_sim:.3f}, mean={mean_sim:.3f}")
    
    if np.isnan(sims).any():
        return {
            "status": "error",
            "message": "❌ Invalid similarity (NaN). Input data not usable."
        }

    # -------------------------
    # GUARD 2: SIMILARITY
    # -------------------------
    if max_sim < SIM_THRESHOLD:
        return {
            "status": "error",
            "message": "❌ No reliable similar patient found (low similarity)."
        }

    # -------------------------
    # TOP-K NEIGHBORS
    # -------------------------
    top_idx = np.argsort(sims)[-TOP_K:][::-1]
    top_sims = sims[top_idx]
    top_matrix = matrix[top_idx]

    print("👉 Top sims:", top_sims)

    # -------------------------
    # WEIGHTED AVERAGE
    # -------------------------
    weights = top_sims / (np.sum(top_sims) + 1e-8)
    estimated = np.average(top_matrix, axis=0, weights=weights)

    # -------------------------
    # SMART COMPLETION
    # -------------------------
    completed = {}

    for i, f in enumerate(feature_order):

        # keep original values
        if f in features:
            completed[f] = float(features[f])
            continue

        neighbor_values = top_matrix[:, i]
        std = np.std(neighbor_values)

        if std < 0.2:
            completed[f] = float(estimated[i])
        else:
            continue

    # -------------------------
    # FINAL CHECK
    # -------------------------
    if len(completed) <= num_known:
        return {
            "status": "error",
            "message": "❌ Unable to reliably complete missing features."
        }

    # -------------------------
    # OUTPUT
    # -------------------------
    return {
        "status": "ok",
        "data": {
            "features": completed,
            "meta": {
                "num_known": num_known,
                "num_completed": len(completed) - num_known,
                "max_similarity": max_sim
            }
        }
    }