import boto3
from config.settings import *
import joblib
import io


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


def validate(features: dict):

    errors = []

    scma = get_schema()

    for f, val in features.items():

        if f not in scma:
            errors.append(f"{f} not in schema")
            continue

        if val is None:
            continue

        try:
            float(val)
        except:
            errors.append(f"{f} must be numeric")

    if errors:
        return {
            "status": "error",
            "message": errors
        }

    return {
        "status": "ok",
        "data": {
            "features": features
        }
    }