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


SCHEMA = None

def get_schema():
    global SCHEMA
    if SCHEMA is None:
        SCHEMA = load_schema()
    return SCHEMA

# Json Extactor
def extract_json(text: str):

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    json_str = match.group(0)

    return json.loads(json_str)


# Main Parser
def parse(text: str):

    schema = get_schema()
    feature_names = list(schema.keys())

    prompt = f"""
    You are a strict JSON generator.

    Extract features ONLY from this EXACT list:
    {feature_names}

    Rules:
    - ONLY use feature names from the list above
    - DO NOT invent new feature names
    - DO NOT use generic names like "gender" or "ethnicity"
    - If a feature is not in the list, ignore it
    - Output ONLY valid JSON
    - Values must be numeric (float)

    Example:
    {{
    "gender_male": 1,
    "ethnicity_asian": 1,
    "age": 65
    }}

    Text:
    {text}
    """

    response = call_llm(prompt)

    print("👉 RAW LLM RESPONSE:", response)

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