import boto3
import json
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


SCHEMA = load_schema()

def validate(features: dict):

    errors = []

    for f, val in features.items():

        # check schema existence
        if f not in SCHEMA:
            errors.append(f"{f} not in schema")
            continue

        if val is None:
            continue

        try:
            float(val)
        except:
            errors.append(f"{f} must be numeric")

    return errors