import json
from agent.llm import call_llm

def plan(user_input, features):

    prompt = f"""
        You are a medical AI agent.

        Available tools:
        1. fetch_patient_tool → get patient features using patient_id
        2. predict_tool → predicts survival risk using features

        Rules:
        - If user provides patient ID → use fetch_patient
        - If features are available → use predict_tool
        - Always return valid JSON
        - Do not explain anything
        - Choose ONE action

        Format:
        {{
        "action": "...",
        "input": {{
            ...
        }}
        }}

        User query:
        {user_input}

        Current features:
        {features}
        """
    
    response = call_llm(prompt)

    try:
        parsed = json.loads(response)

        if "action" not in parsed or "input" not in parsed:
            raise ValueError("Invalid structure")

        return parsed

    except Exception:
        raise ValueError(f"Invalid JSON from LLM:\n{response}")