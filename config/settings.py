import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL")
GENOMIC_FG = os.getenv("GENOMIC_FG")
CLINICAL_FG = os.getenv("CLINICAL_FG")
IMAGING_FG = os.getenv("IMAGING_FG")
BUCKET = os.getenv("BUCKET")
PREFIX = os.getenv("PREFIX")
