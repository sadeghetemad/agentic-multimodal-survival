import boto3
import json
from config.settings import SAGEMAKER_ENDPOINT, AWS_REGION

runtime = boto3.client(service_name = "sagemaker-runtime", region_name = AWS_REGION)


def predict_multimodal(features: dict):
    response = runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(features)
    )

    result = json.loads(response["Body"].read().decode())
    return result