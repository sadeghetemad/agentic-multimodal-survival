import boto3
import json
import re
from config.settings import *
from agent.llm import call_llm
import joblib
import io


# Load Feature Schema
def load_schema():

    s3 = boto3.client("s3", region_name=AWS_REGION)

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=f"{PREFIX}/artifacts/feature_schema.json"
    )

    body = obj["Body"].read()
    schema = joblib.load(io.BytesIO(body))
    return schema


SCHEMA = load_schema()


# Json Extactor
def extract_json(text: str):

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    json_str = match.group(0)

    return json.loads(json_str)


# Main Parser
def parse(text: str):

    feature_names = list(SCHEMA.keys())

    prompt = f"""
        You are a strict JSON generator.

        Extract features ONLY from this list:
        {feature_names}

        Rules:
        - Output ONLY valid JSON
        - No explanation
        - No text before or after JSON
        - Only include features from the list
        - Values must be numeric (float)
        - If a feature is not mentioned, DO NOT include it

        Example:
        {{
        "age": 65,
        "smoking": 1,
        "tumor_size": 4.5
        }}

        Text:
        {text}
        """

    response = call_llm(prompt)

    try:
        parsed = json.loads(response)
    except:
        parsed = extract_json(response)

    cleaned = {}

    for k, v in parsed.items():

        if k not in SCHEMA:
            continue

        try:
            cleaned[k] = float(v)
        except:
            continue

    return cleaned