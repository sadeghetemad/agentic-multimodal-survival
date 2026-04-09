import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL")