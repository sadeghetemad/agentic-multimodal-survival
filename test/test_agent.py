from agent.agent import Agent

def test_agent():
    agent = Agent()

    # user_input = "Predict survival for patient R01-029"
    user_input = "A 65-year-old male with a long history of smoking (30 pack-years) presents with stage III lung cancer. Tumor size is approximately 4.5 cm. No significant comorbidities reported."

    features = {}

    print("\n Running Agent...\n")

    result = agent.run(user_input, features)

    print("\n Agent Output:\n")
    print(result)


if __name__ == "__main__":
    test_agent()