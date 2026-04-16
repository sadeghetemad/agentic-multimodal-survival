# aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 811165582441.dkr.ecr.eu-west-2.amazonaws.com
# docker buildx build --platform linux/arm64 -t 811165582441.dkr.ecr.eu-west-2.amazonaws.com/nsclc-agentic-ai:0.0.1-prod --push .   


import boto3
import json

client = boto3.client("bedrock-agentcore-control", region_name="eu-west-2")

response = client.create_agent_runtime(
    agentRuntimeName="nsclc_survival_predictor_agent",
    roleArn="arn:aws:iam::811165582441:role/service-role/AmazonBedrockAgentCoreRuntimeDefaultServiceRole-3ljll",
    agentRuntimeArtifact={
        "containerConfiguration": {
            "containerUri": "811165582441.dkr.ecr.eu-west-2.amazonaws.com/nsclc-agentic-ai:0.0.1-prod"
        }
    },
    networkConfiguration={
        "networkMode": "PUBLIC"
    },
    environmentVariables={
        "AWS_REGION": "eu-west-2",
        "ENV": "prod",
        "SAGEMAKER_ENDPOINT": "multi-model-health-ml-endpoint",
        "BEDROCK_MODEL": "amazon.nova-micro-v1:0",
        "PYTHONPATH": ".",
        "GENOMIC_FG": "genomic-feature-group-05-19-10-59",
        "CLINICAL_FG": "clinical-feature-group-05-18-48-56",
        "IMAGING_FG": "ct-seg-image-imaging-feature-group",
        "BUCKET": "multimodal-lung-cancer-811165582441-eu-west-2-an",
        "PREFIX": "multi-model-health-ml",
        "MEMORY_ID": "nsclc_short_term_memory-pr9Q6THele",
        "MODEL_THRESHOLD": "0.5"
    }
)

print(json.dumps(response, indent=2, default=str))
print("runtime_id =", response["agentRuntimeId"])
print("status =", response["status"])



