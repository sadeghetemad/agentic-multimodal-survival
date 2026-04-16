from bedrock_agentcore.runtime import BedrockAgentCoreApp
import uuid

# -------------------------
# INIT APP
# -------------------------
app = BedrockAgentCoreApp()

graph = None

def get_graph():
    global graph
    if graph is None:
        print("🧠 Building graph...")
        from agent.graph import build_graph
        graph = build_graph()
    return graph

# -------------------------
# ENTRYPOINT
# -------------------------
@app.entrypoint
def handler(payload, context):
    """
    Entry point for the NSCLC survival prediction agent.

    This function receives user input, routes it through the LangGraph-based
    agent pipeline, and returns a structured prediction response.
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
        g = get_graph()

        result = g.invoke(
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
    

if __name__ == "__main__":
    app.run(port=8080)