from tools.predict_tool import predict_multimodal
from tools.fetch_patient_tool import get_patient_features
from tools.parse_features_tool import parse_features
from tools.complete_features_tool import complete_features
from tools.feature_validator_tool import validate_features


# Predict
def predict_tool(input_data):

    features = input_data.get("features")

    if not features:
        return {
            "status": "error",
            "message": "Missing features for prediction"
        }

    return predict_multimodal(features)


# Fetch Patient
def fetch_patient_tool(input_data):

    patient_id = input_data.get("patient_id")

    if not patient_id:
        return {
            "status": "error",
            "message": "Missing patient_id"
        }

    return get_patient_features(patient_id)


# Parser
def parse_features_tool(input_data):

    if isinstance(input_data, str):
        input_data = {"text": input_data}

    elif isinstance(input_data, dict) and "text" not in input_data:
        input_data["text"] = input_data.get("user_input", "")

    text = input_data.get("text")

    if not text:
        return {
            "status": "error",
            "message": "Missing text input"
        }

    return parse_features(input_data)


# Validator
def validate_features_tool(input_data):

    features = input_data.get("features")

    if not features:
        return {
            "status": "error",
            "message": "Missing features"
        }

    return validate_features(input_data)


# Completer Feature
def complete_features_tool(input_data):

    features = input_data.get("features")

    if not features:
        return {
            "status": "error",
            "message": "Missing features"
        }

    return complete_features(input_data)


# Out of scope
def out_of_scope_tool(input_data):
    return {
        "status": "ok",
        "message": input_data.get("message")
    }


# Tool Registery
TOOLS = {
    "predict_tool": predict_tool,
    "fetch_patient_tool": fetch_patient_tool,
    "parse_features": parse_features_tool,
    "validate_features": validate_features_tool,
    "complete_features": complete_features_tool,
    "out_of_scope": out_of_scope_tool
}