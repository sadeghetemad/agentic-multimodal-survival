from agent.agent import Agent

def test_agent():
    agent = Agent()

    user_input = "Predict survival for patient R01-029"

    features = {}

    print("\n🚀 Running Agent...\n")

    result = agent.run(user_input, features)

    print("\n✅ Agent Output:\n")
    print(result)


if __name__ == "__main__":
    test_agent()