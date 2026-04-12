import boto3
import numpy as np


class PredictionService:

    def __init__(self, endpoint_name: str, region: str):
        self.endpoint_name = endpoint_name
        self.client = boto3.client("sagemaker-runtime", region_name=region)

    # Main Predict
    def predict(self, features):

        
        # Normalize input shape
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
        elif isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        else:
            raise ValueError("Features must be numpy array or list")

        if features.size == 0:
            raise ValueError("Empty feature vector")

        
        # Build CSV payload
        rows = []
        for row in features:
            rows.append(",".join(map(str, row)))

        payload = "\n".join(rows)

        # Call endpoint
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="text/csv",
                Body=payload
            )
        except Exception as e:
            raise RuntimeError(f"SageMaker endpoint call failed: {str(e)}")

        # Parse response
        raw = response["Body"].read().decode("utf-8").strip()

        print("RAW RESPONSE:", raw)

        try:
            value = float(raw.splitlines()[0])
        except Exception:
            raise ValueError(f"Invalid prediction response: {raw}")

        return value