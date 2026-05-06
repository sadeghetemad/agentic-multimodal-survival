import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")
ENV = os.getenv("ENV", "dev")

SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL")

GENOMIC_FG = os.getenv("GENOMIC_FG")
CLINICAL_FG = os.getenv("CLINICAL_FG")
IMAGING_FG = os.getenv("IMAGING_FG")

BUCKET = os.getenv("BUCKET")
PREFIX = os.getenv("PREFIX")

MODEL_THRESHOLD = os.getenv("MODEL_THRESHOLD", "0.5")

# -------------------------
# MEMORY
# -------------------------
MEMORY_ID = os.getenv("MEMORY_ID")
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_SUMMARY_TEMPLATE = os.getenv("MEMORY_SUMMARY_TEMPLATE", "/clinicians/{actor_id}/sessions/{session_id}")
MEMORY_PREFERENCES_TEMPLATE = os.getenv("MEMORY_PREFERENCES_TEMPLATE","/clinicians/{actor_id}/preferences")

# -------------------------
# TOOL BACKEND
# local | gateway
# -------------------------
TOOL_PROVIDER = os.getenv("TOOL_PROVIDER", "local").lower()

# -------------------------
# AGENTCORE GATEWAY
# -------------------------
GATEWAY_URL = os.getenv("GATEWAY_URL")  # full MCP url, e.g. https://.../mcp
GATEWAY_AUTH_TYPE = os.getenv("GATEWAY_AUTH_TYPE", "none").lower()  # none | bearer | api_key
GATEWAY_BEARER_TOKEN = os.getenv("GATEWAY_BEARER_TOKEN")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")

# Tool names in gateway are usually target_name___tool_name
GATEWAY_PARSE_FEATURES_TOOL = os.getenv("GATEWAY_PARSE_FEATURES_TOOL")
GATEWAY_VALIDATE_FEATURES_TOOL = os.getenv("GATEWAY_VALIDATE_FEATURES_TOOL")
GATEWAY_COMPLETE_FEATURES_TOOL = os.getenv("GATEWAY_COMPLETE_FEATURES_TOOL")
GATEWAY_FETCH_PATIENT_TOOL = os.getenv("GATEWAY_FETCH_PATIENT_TOOL")
GATEWAY_PREDICT_TOOL = os.getenv("GATEWAY_PREDICT_TOOL")