from langchain.tools import tool
from tools.predict import predict_multimodal
import json

@tool
def predict_tool(input_data: str):
    """Predict survival risk given patient features as JSON"""
    data = json.loads(input_data)
    return predict_multimodal(data)

