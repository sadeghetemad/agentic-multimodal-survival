import json
from agent.llm import call_llm

def plan(user_input, features):
    prompt = f"""
        You are an AI medical agent.

        Available tools:
        - predict_tool
        - simulate_change
        - explain_prediction

        Decide the best action.

        Return ONLY valid JSON:
        {{
        "action": "...",
        "input": ...
        }}

        User query: {user_input}
        Patient data: {features}
        """

    response = call_llm(prompt)

    try:
        return json.loads(response)
    except:
        raise ValueError(f"Invalid JSON from LLM: {response}")