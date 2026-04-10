from tools.predict_tool import predict_multimodal
from tools.fetch_patient_tool import get_patient_features


def predict_tool(input_data):
    return predict_multimodal(input_data["features"])


def fetch_patient_tool(input_data):
    return get_patient_features(input_data["patient_id"])


TOOLS = {
    "predict_tool": predict_tool,
    "fetch_patient_tool": fetch_patient_tool
}