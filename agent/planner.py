import json
from agent.llm import call_llm


def detect_intent(text: str) -> str:

    prompt = f"""
        You are an intent classifier.

        Classify the user input into one of these categories:
        - MEDICAL → if it contains patient info, cancer, diagnosis, clinical data
        - OUT_OF_SCOPE → greetings, casual talk, unrelated topics

        Rules:
        - Output ONLY one word
        - No explanation

        Examples:
        Input: "hi"
        Output: OUT_OF_SCOPE

        Input: "65 year old smoker with lung cancer"
        Output: MEDICAL

        Input:
        {text}

        Output:
        """

    response = call_llm(prompt).strip().upper()

    if response not in ["MEDICAL", "OUT_OF_SCOPE"]:
        return "OUT_OF_SCOPE"

    return response


def plan(user_input, features):

    intent = detect_intent(user_input)

    if intent == "OUT_OF_SCOPE":
        return [{
            "action": "out_of_scope",
            "input": {
                "message": "I am a medical AI specialized in lung cancer prediction. Please provide relevant clinical information."
            }
        }]

    prompt = f"""
        You are a medical AI agent for NSCLC survival prediction.

        Available tools:
        1. fetch_patient_tool
        2. parse_features
        3. validate_features
        4. complete_features
        5. predict_tool

        Rules:

        - If user provides patient_id:
            return:
            [
            {{"action": "fetch_patient_tool", "input": {{"patient_id": "..."}}}},
            {{"action": "predict_tool", "input": {{}}}}
            ]

        - If user provides free text:
            return:
            [
            {{"action": "parse_features", "input": {{}}}},
            {{"action": "validate_features", "input": {{}}}},
            {{"action": "complete_features", "input": {{}}}},
            {{"action": "predict_tool", "input": {{}}}}
            ]

        - If features are already complete:
            return:
            [
            {{"action": "predict_tool", "input": {{}}}}
            ]

        - Always return ONLY JSON list
        - No explanation
        - Do NOT return text outside JSON

        User input:
        {user_input}

        Current features:
        {features}
        """

    response = call_llm(prompt)

    try:
        parsed = json.loads(response)

        if not isinstance(parsed, list):
            raise ValueError("Planner must return list")

        for step in parsed:
            if "action" not in step:
                raise ValueError("Invalid step structure")

        return parsed

    except Exception:
        raise ValueError(f"Invalid JSON from LLM:\n{response}")
    