from bedrock_agentcore.runtime import BedrockAgentCoreApp
import uuid

from agent.graph import build_graph
from agent.clinical_memory_service import ClinicalMemoryService
from services.prediction_pipeline import init_pipeline


app = BedrockAgentCoreApp()

print("🔥 Warming AgentCore app...")

graph = build_graph()
memory_service = ClinicalMemoryService()

try:
    init_pipeline()
except Exception as e:
    print(f"⚠️ Pipeline warmup failed: {e}")

print("✅ AgentCore app is ready")


@app.entrypoint
def handler(payload, context):
    user_input = payload.get("input", "")
    session_id = payload.get("session_id") or str(uuid.uuid4())
    actor_id = payload.get("actor_id", "default-user")

    if not user_input:
        return {
            "status": "error",
            "message": "Missing input"
        }

    try:
        result = graph.invoke(
            {
                "input": user_input,
                "actor_id": actor_id,
                "session_id": session_id
            },
            config={
                "configurable": {
                    "thread_id": session_id,
                    "actor_id": actor_id,
                }
            }
        )

        response_text = result.get("response", "")
        patient_id = result.get("patient_id")

        memory_service.capture_turn(
            actor_id=actor_id,
            session_id=session_id,
            user_text=user_input,
            assistant_text=response_text
        )

        return {
            "status": "ok",
            "session_id": session_id,
            "actor_id": actor_id,
            "patient_id": patient_id,
            "response": response_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    app.run(port=8080)