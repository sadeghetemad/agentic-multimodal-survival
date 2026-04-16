import boto3
import joblib
import numpy as np
import os
import pandas as pd
import xgboost as xgb
import shap

from services.prediction_service import PredictionService
from config.settings import *
from agent.llm import call_llm


# =========================
# S3 Artifact Loader
# =========================
def load_artifact_from_s3(filename):

    local_path = os.path.join("artifacts", filename)

    if not os.path.exists(local_path):
        print(f"⬇️ Downloading {filename} from S3...")

        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/{filename}",
            local_path
        )

    return joblib.load(local_path)


# =========================
# Load Model
# =========================
def load_raw_model(filename):

    local_path = os.path.join("artifacts", filename)

    if not os.path.exists(local_path):
        print("⬇️ Downloading model from S3...")

        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.download_file(
            BUCKET,
            f"{PREFIX}/artifacts/xgboost-model",
            local_path
        )

    print("⚙️ Loading model...")

    model = xgb.XGBClassifier()
    model.load_model(local_path)

    print("✅ Model loaded")

    return model



predictor = PredictionService(
    endpoint_name=SAGEMAKER_ENDPOINT,
    region=AWS_REGION
)


scaler = None
pca = None
feature_order = None
model = None
explainer = None
predictor = None


def init_pipeline():
    global scaler, pca, feature_order, model, explainer, predictor

    if model is not None:
        return

    print("🔥 Lazy loading ML pipeline...")

    scaler = load_artifact_from_s3("scaler.joblib")
    pca = load_artifact_from_s3("pca.joblib")
    feature_order = load_artifact_from_s3("feature_order.joblib")
    model = load_raw_model("xgboost-model")

    predictor = PredictionService(
        endpoint_name=SAGEMAKER_ENDPOINT,
        region=AWS_REGION
    )

    try:
        explainer = shap.TreeExplainer(model)
        print("✅ SHAP ready")
    except Exception as e:
        print("⚠️ SHAP failed:", e)
        explainer = None

    print("🚀 ML ready")



# =========================
# SHAP Explanation
# =========================
def compute_shap_explanation(X_pca, features):

    if explainer is None:
        return []

    shap_values = explainer.shap_values(X_pca)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_original = np.dot(
        shap_values,
        pca.components_
    )

    feature_contrib = dict(zip(feature_order, shap_original[0]))

    sorted_features = sorted(
        feature_contrib.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top = [
        {
            "feature": name,
            "value": float(features.get(name, 0)),
            "contribution": float(val)
        }
        for name, val in sorted_features[:10]
    ]

    return top


# =========================
# LLM Explanation
# =========================
def explain_with_llm(prob, risk, feature_explanations):

    if not feature_explanations:
        return "No explanation available."

    text = "\n".join([
        f"{f['feature']} = {f['value']} (contribution={f['contribution']:.3f})"
        for f in feature_explanations
    ])

    prompt = f"""
    You are a clinical AI assistant specializing in NSCLC survival prediction (NOT diagnosis).

    Prediction:
    - Risk level: {risk}
    - Probability of mortality: {prob:.3f}

    Top contributing features:
    {text}

    Instructions:
    - Focus ONLY on mortality risk
    - Identify at least 3 key features
    - Positive contribution → increases risk
    - Negative contribution → decreases risk
    - Be cautious and concise
    """

    response = call_llm(prompt)

    return response or "No explanation generated."


# =========================
# Main Prediction Pipeline
# =========================
def predict_multimodal(features: dict):

    init_pipeline()

    if not features:
        return {
            "status": "error",
            "message": "Empty features"
        }

    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_order)
    df = df.fillna(0)

    if df.shape[1] != len(feature_order):
        return {
            "status": "error",
            "message": f"Feature mismatch: expected {len(feature_order)}, got {df.shape[1]}"
        }

    try:
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Preprocessing failed: {str(e)}"
        }

    try:
        result = predictor.predict(X_pca)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

    prob = result.get("probability") if isinstance(result, dict) else float(result)
    risk = "high" if prob > float(MODEL_THRESHOLD) else "low"

    feature_explanations = compute_shap_explanation(
        X_pca,
        features
    )

    llm_explanation = explain_with_llm(
        prob,
        risk,
        feature_explanations
    )

    return {
        "status": "ok",
        "probability": float(prob),
        "risk": risk,
        "threshold": float(MODEL_THRESHOLD),
        "top_features": feature_explanations,
        "analysis": llm_explanation
    }