import json
import re
from agent.llm import call_llm


def detect_intent(text: str) -> str:

    prompt = f"""
        You are a strict intent classifier for a medical AI system focused on lung cancer (NSCLC).

        Your task is to classify the user input into ONE of the following categories:

        1. MEDICAL
        2. OUT_OF_SCOPE

        Rules:
        - Output ONLY one word: MEDICAL or OUT_OF_SCOPE
        - Be conservative: prefer MEDICAL if uncertain

        Input:
        {text}

        Output:
        """

    response = call_llm(prompt).strip().upper()

    if response not in ["MEDICAL", "OUT_OF_SCOPE"]:
        return "OUT_OF_SCOPE"

    return response


def extract_patient_id(text: str):
    match = re.search(r"(R\d{2}-\d+|patient[_\-]?\d+|case[_\-]?\d+)", text, re.IGNORECASE)
    return match.group(0) if match else None


def plan(user_input, features):

    intent = detect_intent(user_input)

    patient_id = extract_patient_id(user_input)
    if patient_id:
        return [
            {"action": "fetch_patient_tool", "input": {"patient_id": patient_id}},
            {"action": "predict_tool", "input": {}}
        ]


    if intent == "OUT_OF_SCOPE":
        prompt = f"""
        You are a medical AI specialized in lung cancer prediction.

        The user asked something outside your domain.

        Politely explain that their request is NOT related to lung cancer prediction.

        User input:
        "{user_input}"

        Constraints:
        - Be short (2-3 sentences)
        - Be polite but firm
        """

        llm_response = call_llm(prompt)

        return [{
            "action": "out_of_scope",
            "input": {
                "message": llm_response
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

        - Do NOT invent tools
        - Always return ONLY JSON list
        - No explanation

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