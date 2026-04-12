from langgraph.graph import StateGraph, END
from typing import TypedDict, List

from agent.planner import plan
from agent.executor import execute_tool

from tools.predict_tool import predict_multimodal


# Agent State
class AgentState(TypedDict):
    user_input: str
    features: dict
    plan: List[dict]
    tool_result: dict
    response: str


# Output Formatter
def format_prediction_output(pred):

    features_text = "\n".join([
        f"- {f['feature']} = {f['value']} (importance={f['importance']:.3f})"
        for f in pred.get("top_features", [])
    ])

    return f"""
            Prediction Result
            -----------------
            Risk: {pred['risk']}
            Probability: {pred['probability']:.3f}

            Top Contributing Features:
            {features_text if features_text else "No important features found"}

            AI Analysis:
            {pred.get('analysis', 'No analysis available')}
            """


# Nodes
def planner_node(state):

    plan_output = plan(state["user_input"], state.get("features", {}))

    return {"plan": plan_output}


def tool_node(state):

    result = execute_tool(state["plan"], state)

    return {"tool_result": result}


def response_node(state):

    result = state["tool_result"]

    
    if result.get("status") == "error":
        return {"response": f"Error: {result['message']}"}
    
    
    if result.get("status") == "ok" and "message" in result:
        return {
            "response": result["message"]
    }

    if result.get("status") == "ok" and "features" in result and "risk" not in result:

        pred = predict_multimodal(result["features"])

        return {
            "response": format_prediction_output(pred)
        }

    if result.get("status") == "ok" and "risk" in result:

        return {
            "response": format_prediction_output(result)
        }

    return {"response": f"Unexpected result: {result}"}


# Graph
def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_node("respond", response_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "tool")
    graph.add_edge("tool", "respond")
    graph.add_edge("respond", END)

    return graph.compile()