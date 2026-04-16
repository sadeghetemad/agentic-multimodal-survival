from app.agentcore_app import handler
import uuid


session_id = str(uuid.uuid4())

def run_query(user_input: str):
    response = handler(
        {
            "input": user_input,
            "session_id": session_id
        },
        None
    )
    return response.get("response", "")


# CLI LOOP
if __name__ == "__main__":

    print("🫁 NSCLC Agent (CLI Mode)")
    print("Type 'exit' to quit\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        try:
            output = run_query(user_input)
            print("\nAI:", output, "\n")

        except Exception as e:
            print("Error:", str(e))