from aws_agentcore.app import handler

def run_agent(input_text, session_id=None):
    payload = {
        "input": input_text
    }

    if session_id:
        payload["session_id"] = session_id

    return handler(payload, None)