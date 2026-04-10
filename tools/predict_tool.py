import boto3
import joblib
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from services.prediction_service import PredictionService
from config.settings import *
from agent.llm import call_llm


# ==========================
# S3 ARTIFACT LOADER
# ==========================

def load_artifact_from_s3(filename):

    local_path = filename

    if not os.path.exists(local_path):
        s3 = boto3.client("s3", region_name=AWS_REGION)

        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/{filename}",
            local_path
        )

    return joblib.load(local_path)


# ==========================
# LOAD MODEL
# ==========================
def load_raw_model():
    filename = "xgboost-model" 

    if not os.path.exists(filename):

        print("Downloading model from S3...")

        s3 = boto3.client("s3", region_name=AWS_REGION)

        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/{filename}",
            filename
        )

    print("Loading model...")

    booster = xgb.Booster()
    booster.load_model(filename)

    print("Model loaded successfully")

    return booster


# ==========================
# LOAD ARTIFACTS
# ==========================

scaler = load_artifact_from_s3("scaler.joblib")
pca = load_artifact_from_s3("pca.joblib")
feature_order = load_artifact_from_s3("feature_order.joblib")

model = load_raw_model()

predictor = PredictionService(
    endpoint_name=SAGEMAKER_ENDPOINT,
    region=AWS_REGION
)

THRESHOLD = 0.5


# ==========================
# FEATURE IMPORTANCE (PCA AWARE)
# ==========================

def compute_feature_importance():

    score = model.get_score(importance_type="gain")

    if not score:
        return []

    pc_importance = np.zeros(len(pca.components_))

    for k, v in score.items():
        idx = int(k.replace("f", ""))
        pc_importance[idx] = v

    feature_importance = np.dot(
        pc_importance,
        np.abs(pca.components_)
    )

    importance_dict = dict(zip(feature_order, feature_importance))

    return sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )


# ==========================
# BUILD EXPLANATION (LOCAL)
# ==========================

def build_explanation(features):

    importance = compute_feature_importance()

    top = []

    for name, score in importance[:10]:
        val = float(features.get(name, 0))

        if val != 0:
            top.append({
                "feature": name,
                "value": val,
                "importance": float(score)
            })

    return top


# ==========================
# LLM EXPLANATION
# ==========================

def explain_with_llm(prob, risk, feature_explanations):

    text = "\n".join([
        f"{f['feature']} = {f['value']} (importance={f['importance']:.3f})"
        for f in feature_explanations
    ])

    prompt = f"""
        You are a medical AI assistant.

        Prediction:
        - Risk: {risk}
        - Probability: {prob:.3f}

        Important patient features:
        {text}

        Explain:
        - Why this patient is {risk} risk
        - Which features contributed most
        - Use simple clinical reasoning
        - Be concise
        """

    return call_llm(prompt)


# ==========================
# MAIN FUNCTION
# ==========================

def predict_multimodal(features: dict):

    # -------------------------
    # Validation
    # -------------------------
    if not features:
        return {
            "status": "error",
            "message": "Empty features"
        }

    missing = [f for f in feature_order if f not in features]
    if missing:
        return {
            "status": "error",
            "message": f"Missing features: {missing}"
        }

    try:
        values = [float(features[f]) for f in feature_order]
    except Exception as e:
        return {
            "status": "error",
            "message": f"Invalid feature values: {str(e)}"
        }

    # -------------------------
    # Transform
    # -------------------------

    df = pd.DataFrame([values], columns=feature_order)

    try:
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Preprocessing failed: {str(e)}"
        }

    # -------------------------
    # Predict (Endpoint)
    # -------------------------
    try:
        prob = predictor.predict(X_pca[0])
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

    risk = "high" if prob > THRESHOLD else "low"

    # -------------------------
    # Explain
    # -------------------------

    feature_explanations = build_explanation(features)

    llm_explanation = explain_with_llm(
        prob,
        risk,
        feature_explanations
    )

    # -------------------------
    # Output
    # -------------------------
    return {
        "status": "ok",
        "probability": float(prob),
        "risk": risk,
        "threshold": THRESHOLD,
        "top_features": feature_explanations,
        "analysis": llm_explanation
    }