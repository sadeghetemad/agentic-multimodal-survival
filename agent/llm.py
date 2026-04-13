from langchain_aws import ChatBedrock
from config.settings import BEDROCK_MODEL, AWS_REGION

llm = ChatBedrock(
    model=BEDROCK_MODEL,
    region=AWS_REGION,
    model_kwargs={
        "temperature": 0.0
    }
)


def call_llm(prompt: str):
    response = llm.invoke(prompt)
    return response.content