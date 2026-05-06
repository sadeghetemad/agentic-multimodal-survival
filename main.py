from app.agentcore_app import handler
import uuid


# -------------------------
# SESSION + USER
# -------------------------
session_id = str(uuid.uuid4())
actor_id = "clinician_local_1"


def run_query(user_input: str):
    response = handler(
        {
            "input": user_input,
            "session_id": session_id,
            "actor_id": actor_id
        },
        None
    )

    if response.get("status") != "ok":
        return f"❌ {response.get('message')}"

    return response.get("response", "")


# -------------------------
# CLI LOOP
# -------------------------
if __name__ == "__main__":

    print("🫁 NSCLC Agent (CLI Mode)")
    print("Commands:")
    print("  /new      -> start new session")
    print("  /who      -> show actor_id")
    print("  /session  -> show session_id")
    print("  exit      -> quit\n")

    while True:

        user_input = input("You: ").strip()

        # -------------------------
        # EXIT
        # -------------------------
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        # -------------------------
        # NEW SESSION (VERY IMPORTANT)
        # -------------------------
        if user_input.lower() == "/new":
            session_id = str(uuid.uuid4())
            print(f"🔄 New session started: {session_id}")
            continue

        # -------------------------
        # DEBUG INFO
        # -------------------------
        if user_input.lower() == "/who":
            print(f"👤 actor_id: {actor_id}")
            continue

        if user_input.lower() == "/session":
            print(f"🧵 session_id: {session_id}")
            continue

        # -------------------------
        # RUN QUERY
        # -------------------------
        try:
            output = run_query(user_input)
            print("\nAI:", output, "\n")

        except Exception as e:
            print("❌ Error:", str(e))