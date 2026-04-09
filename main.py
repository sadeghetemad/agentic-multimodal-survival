from agent.graph import build_graph

graph = build_graph()

input_data = {
    "user_input": "Predict survival for age 65 and tumor size 3.2"
}

result = graph.invoke(input_data)

print(result["response"])