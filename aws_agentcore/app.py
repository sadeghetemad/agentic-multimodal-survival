from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent.graph import build_graph
import uuid

# -------------------------
# INIT APP
# -------------------------
app = BedrockAgentCoreApp()

graph = build_graph()

# -------------------------
# ENTRYPOINT
# -------------------------
@app.entrypoint
def handler(payload, context):
    """
    Entry point for the NSCLC survival prediction agent.

    This function receives user input, routes it through the LangGraph-based
    agent pipeline, and returns a structured prediction response.

    Parameters
    ----------
    payload : dict
        Request payload containing user input and optional session metadata.

        Expected format:
        {
            "input": str,
                Natural language query or patient identifier.
                Examples:
                - "R01-029"
                - "patient with id R01-100"
                - "age 65 smoker male stage 3"

            "session_id": str, optional
                Unique identifier for conversation/session tracking.
                If not provided, a UUID will be generated automatically.
        }

    context : object
        Runtime context provided by AgentCore (not used directly).

    Returns
    -------
    dict
        Structured response containing prediction results or error.

        Success:
        {
            "status": "ok",
            "session_id": str,
            "response": str
        }

        Error:
        {
            "status": "error",
            "message": str
        }

    Behavior
    --------
    - Detects whether input refers to a patient ID or raw clinical text
    - Executes the appropriate processing pipeline:
        * Patient flow:
            fetch_patient → predict
        * Text flow:
            parse → validate → complete → predict
    - Generates survival risk prediction and explanation
    - Maintains session continuity via AgentCore memory

    Notes
    -----
    - Prediction is for mortality risk estimation, NOT diagnosis
    - Output should be interpreted as probabilistic guidance
    - Missing or invalid inputs will return structured errors
    """

    # -------------------------
    # INPUT
    # -------------------------
    user_input = payload.get("input", "")
    session_id = payload.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    if not user_input:
        return {
            "status": "error",
            "message": "Missing input"
        }

    try:
        result = graph.invoke(
                {
                    "input": user_input
                },
                config={
                    "configurable": {
                        "thread_id": session_id,
                        "actor_id": "nsclc-agent"
                    }
                }
            )

        response_text = result.get("response", "")

        return {
            "status": "ok",
            "session_id": session_id,
            "response": response_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }