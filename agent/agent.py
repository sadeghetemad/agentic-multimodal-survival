from agent.graph import build_graph


class Agent:

    def __init__(self):
        self.graph = build_graph()

    def run(self, user_input, features=None):

        state = {
            "user_input": user_input,
            "features": features or {},
            "plan": [],
            "tool_result": {},
            "response": ""
        }

        result = self.graph.invoke(state)

        return result.get("response", "No response generated")