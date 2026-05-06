from abc import ABC, abstractmethod
from typing import Dict, Any


class ToolProvider(ABC):
    @abstractmethod
    def parse_features(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def complete_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fetch_patient(self, patient_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError