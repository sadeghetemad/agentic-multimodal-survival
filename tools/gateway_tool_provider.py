import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

from config.settings import (
    GATEWAY_URL,
    GATEWAY_AUTH_TYPE,
    GATEWAY_BEARER_TOKEN,
    GATEWAY_API_KEY,
    GATEWAY_PARSE_FEATURES_TOOL,
    GATEWAY_VALIDATE_FEATURES_TOOL,
    GATEWAY_COMPLETE_FEATURES_TOOL,
    GATEWAY_FETCH_PATIENT_TOOL,
    GATEWAY_PREDICT_TOOL,
)
from tools.tool_provider import ToolProvider
from tools.local_tool_provider import LocalToolProvider


class GatewayToolProvider(ToolProvider):
    """
    Gateway-backed provider.
    If a tool name is missing, we gracefully fall back to local provider.
    """

    def __init__(self):
        self.gateway_url = GATEWAY_URL
        self.auth_type = GATEWAY_AUTH_TYPE
        self.bearer_token = GATEWAY_BEARER_TOKEN
        self.api_key = GATEWAY_API_KEY
        self.local_fallback = LocalToolProvider()

        self.tool_names = {
            "parse_features": GATEWAY_PARSE_FEATURES_TOOL,
            "validate_features": GATEWAY_VALIDATE_FEATURES_TOOL,
            "complete_features": GATEWAY_COMPLETE_FEATURES_TOOL,
            "fetch_patient": GATEWAY_FETCH_PATIENT_TOOL,
            "predict": GATEWAY_PREDICT_TOOL,
        }

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json"
        }

        if self.auth_type == "bearer" and self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        if self.auth_type == "api_key" and self.api_key:
            headers["x-api-key"] = self.api_key

        return headers

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.gateway_url:
            raise ValueError("GATEWAY_URL is not configured")

        req = urllib.request.Request(
            self.gateway_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gateway HTTPError {e.code}: {body}")
        except Exception as e:
            raise RuntimeError(f"Gateway request failed: {str(e)}")

    def list_tools(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": "list-tools-request",
            "method": "tools/list"
        }
        return self._post_json(payload)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": "call-tool-request",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        response = self._post_json(payload)
        return self._normalize_tool_response(response)

    def _normalize_tool_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in response:
            return {
                "status": "error",
                "message": response["error"]
            }

        result = response.get("result", {})

        # MCP content may come in various shapes
        structured_content = result.get("structuredContent")
        if isinstance(structured_content, dict):
            return structured_content

        content = result.get("content", [])
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict):
                if "json" in first and isinstance(first["json"], dict):
                    return first["json"]

                if "text" in first:
                    text = first["text"]
                    try:
                        return json.loads(text)
                    except Exception:
                        return {
                            "status": "ok",
                            "data": {
                                "text": text
                            }
                        }

        if isinstance(result, dict):
            return result

        return {
            "status": "error",
            "message": "Unknown gateway tool response shape"
        }

    def _call_or_fallback(
        self,
        logical_tool_name: str,
        fallback_fn,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        gateway_tool_name = self.tool_names.get(logical_tool_name)

        if not gateway_tool_name:
            return fallback_fn(**arguments)

        try:
            return self.call_tool(gateway_tool_name, arguments)
        except Exception as e:
            print(f"[GatewayToolProvider] fallback for {logical_tool_name}: {e}")
            return fallback_fn(**arguments)

    def parse_features(self, text: str) -> Dict[str, Any]:
        return self._call_or_fallback(
            "parse_features",
            self.local_fallback.parse_features,
            {"text": text}
        )

    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self._call_or_fallback(
            "validate_features",
            self.local_fallback.validate_features,
            {"features": features}
        )

    def complete_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self._call_or_fallback(
            "complete_features",
            self.local_fallback.complete_features,
            {"features": features}
        )

    def fetch_patient(self, patient_id: str) -> Dict[str, Any]:
        return self._call_or_fallback(
            "fetch_patient",
            self.local_fallback.fetch_patient,
            {"patient_id": patient_id}
        )

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self._call_or_fallback(
            "predict",
            self.local_fallback.predict,
            {"features": features}
        )