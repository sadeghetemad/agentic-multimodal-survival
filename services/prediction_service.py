import boto3
import numpy as np


class PredictionService:

    def __init__(self, endpoint_name: str, region: str):
        self.endpoint_name = endpoint_name
        self.client = boto3.client("sagemaker-runtime", region_name=region)

    # ======================================================
    # MAIN PREDICT
    # ======================================================

    def predict(self, features):

        # -------------------------
        # Normalize input
        # -------------------------
        if isinstance(features, np.ndarray):
            values = features.flatten().tolist()
        elif isinstance(features, list):
            values = features
        else:
            raise ValueError("Features must be numpy array or list")

        if not values:
            raise ValueError("Empty feature vector")

        # -------------------------
        # Build payload
        # -------------------------
        payload = ",".join(map(str, values))

        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="text/csv",
                Body=payload
            )

        except Exception as e:
            raise RuntimeError(f"SageMaker endpoint call failed: {str(e)}")

        # -------------------------
        # Parse response
        # -------------------------
        raw = response["Body"].read().decode("utf-8").strip()

        try:
            # handle "0.123" or "0.123,..." cases
            value = float(raw.split(",")[0])
        except Exception:
            raise ValueError(f"Invalid prediction response: {raw}")

        return value