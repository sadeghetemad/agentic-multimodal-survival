import boto3
import json
from config.settings import BEDROCK_MODEL, AWS_REGION

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def call_llm(prompt: str):
    
    # Antrophic
    # body = {
    #     "anthropic_version": "bedrock-2023-05-31",
    #     "max_tokens": 500,
    #     "temperature": 0.3,
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt}
    #             ]
    #         }
    #     ]
    # }

      # response = bedrock.invoke_model(
    #     modelId=BEDROCK_MODEL,
    #     body=json.dumps(body)
    # )

    # result = json.loads(response["body"].read())

    # return result["content"][0]["text"]

    
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 500,
            "temperature": 0.3,
            "topP": 0.9
        }
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    return result["output"]["message"]["content"][0]["text"]

  