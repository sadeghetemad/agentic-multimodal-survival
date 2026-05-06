from typing import Dict, Any

from tools.tool_provider import ToolProvider
from tools.langchain_tools import (
    parse_features,
    validate_features,
    complete_features,
    fetch_patient,
    predict
)


class LocalToolProvider(ToolProvider):
    def parse_features(self, text: str) -> Dict[str, Any]:
        return parse_features.invoke({"text": text})

    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return validate_features.invoke({"features": features})

    def complete_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return complete_features.invoke({"features": features})

    def fetch_patient(self, patient_id: str) -> Dict[str, Any]:
        return fetch_patient.invoke({"patient_id": patient_id})

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return predict.invoke({"features": features})