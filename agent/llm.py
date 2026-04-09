import boto3
import json
from config.settings import BEDROCK_MODEL, AWS_REGION


bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def call_llm(prompt: str):
    body = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL,
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]