import json
import os
import sys
from typing import Any, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.prediction_pipeline import predict_multimodal


def _extract_arguments(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gateway/Lambda integration may pass arguments in slightly different shapes.
    """
    if not isinstance(event, dict):
        return {}

    if isinstance(event.get("arguments"), dict):
        return event["arguments"]

    params = event.get("params")
    if isinstance(params, dict) and isinstance(params.get("arguments"), dict):
        return params["arguments"]

    # fallback for direct invocation
    return event


def lambda_handler(event, context):
    print("PREDICT EVENT:", json.dumps(event, default=str))

    try:
        args = _extract_arguments(event)
        features = args.get("features")

        if not isinstance(features, dict) or not features:
            return {
                "status": "error",
                "message": "features are required and must be a non-empty object"
            }

        result = predict_multimodal(features)

        if not isinstance(result, dict):
            return {
                "status": "error",
                "message": "Invalid response from prediction pipeline"
            }

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"predict_lambda failed: {str(e)}"
        }