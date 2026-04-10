from agent.graph import build_graph

class Agent:

    def __init__(self):
        self.graph = build_graph()

    def run(self, user_input, features):

        state = {
            "user_input": user_input,
            "features": features,
            "history": []   # future use
        }

        result = self.graph.invoke(state)

        return result["response"]