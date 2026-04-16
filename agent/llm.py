from langchain_aws import ChatBedrock
from config.settings import BEDROCK_MODEL, AWS_REGION

_llm = None

def get_llm():
    global _llm

    if _llm is None:
        print(" Initializing Bedrock LLM...")
        _llm = ChatBedrock(
            model=BEDROCK_MODEL,
            region=AWS_REGION,
            model_kwargs={
                "temperature": 0.0
            }
        )

    return _llm


def call_llm(prompt: str):
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content