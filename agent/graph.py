from langgraph.graph import StateGraph, END

from agent.planner import plan
from agent.executor import execute_tool

from typing import TypedDict


# ==========================
# STATE
# ==========================

class AgentState(TypedDict):
    user_input: str
    features: dict
    plan: dict
    tool_result: dict
    response: str


# ==========================
# FORMATTER (NEW)
# ==========================

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


# ==========================
# NODES
# ==========================

def planner_node(state):

    plan_output = plan(
        state["user_input"],
        state["features"]
    )

    return {"plan": plan_output}


def tool_node(state):

    result = execute_tool(state["plan"])

    return {"tool_result": result}


def response_node(state):

    result = state["tool_result"]

    # -------------------------
    # ERROR
    # -------------------------
    if result.get("status") == "error":
        return {"response": f"Error: {result['message']}"}

    # -------------------------
    # CASE 1: fetch_patient → needs prediction
    # -------------------------
    if result.get("status") == "ok" and "features" in result:

        from tools.predict_tool import predict_multimodal

        pred = predict_multimodal(result["features"])

        return {
            "response": format_prediction_output(pred)
        }

    # -------------------------
    # CASE 2: already predicted
    # -------------------------
    if result.get("status") == "ok" and "risk" in result:

        return {
            "response": format_prediction_output(result)
        }

    # -------------------------
    # FALLBACK
    # -------------------------
    return {"response": f"Unexpected result: {result}"}


# ==========================
# GRAPH
# ==========================

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