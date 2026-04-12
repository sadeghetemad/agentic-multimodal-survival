import boto3
import joblib
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from services.prediction_service import PredictionService
from config.settings import *
from agent.llm import call_llm


# S3 Artifcat Loader
def load_artifact_from_s3(filename):

    local_path = os.path.join("artifacts", filename)

    if not os.path.exists(local_path):
        os.makedirs("artifacts", exist_ok=True)

        s3 = boto3.client("s3", region_name=AWS_REGION)

        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/{filename}",
            local_path
        )

    return joblib.load(local_path)


# Load model
def load_raw_model():
    local_path = os.path.join("artifacts", "xgboost-model")

    if not os.path.exists(local_path):

        print("Downloading model from S3...")

        s3 = boto3.client("s3", region_name=AWS_REGION)

        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/xgboost-model",
            local_path
        )

    print("Loading model...")

    booster = xgb.Booster()
    booster.load_model(local_path)

    print("Model loaded successfully")

    return booster


# Load Artifacts
scaler = load_artifact_from_s3("scaler.joblib")
pca = load_artifact_from_s3("pca.joblib")
feature_order = load_artifact_from_s3("feature_order.joblib")
model = load_raw_model()

predictor = PredictionService(
    endpoint_name=SAGEMAKER_ENDPOINT,
    region=AWS_REGION
)

THRESHOLD = 0.5


# Feature Importance
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


# Buld Explanator
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


# LLM Explanator
def explain_with_llm(prob, risk, feature_explanations):

    text = "\n".join([
        f"{f['feature']} = {f['value']} (importance={f['importance']:.3f})"
        for f in feature_explanations
    ])

    prompt = f"""
        You are an experienced medical AI assistant specializing in NSCLC diagnosis.

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


# Main 
def predict_multimodal(features: dict):

    # Validation
    if not features:
        return {
            "status": "error",
            "message": "Empty features"
        }

    # Build Dataframe
    df = pd.DataFrame([features])

    # enforce exact training order
    df = df.reindex(columns=feature_order)

    # fill like training
    df = df.fillna(0)

    # sanity check
    if df.shape[1] != len(feature_order):
        return {
            "status": "error",
            "message": f"Feature mismatch: expected {len(feature_order)}, got {df.shape[1]}"
        }

    # Transform
    try:
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Preprocessing failed: {str(e)}"
        }

    print("PCA SHAPE:", X_pca.shape)

    # Predict
    try:
        prob = predictor.predict(X_pca)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

    risk = "high" if prob > THRESHOLD else "low"

    # Explain
    feature_explanations = build_explanation(features)

    llm_explanation = explain_with_llm(
        prob,
        risk,
        feature_explanations
    )

    # Output
    return {
        "status": "ok",
        "probability": float(prob),
        "risk": risk,
        "threshold": THRESHOLD,
        "top_features": feature_explanations,
        "analysis": llm_explanation
    }