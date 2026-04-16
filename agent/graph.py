from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict
from agent.llm import call_llm

import json

from tools.langchain_tools import (
    parse_features,
    validate_features,
    complete_features,
    fetch_patient,
    predict
)


# -------------------------
# STATE
# -------------------------
class AgentState(TypedDict):
    input: str
    route: str
    features: Dict
    patient_id: str 
    prediction: Dict
    response: str
    valid: bool


# -------------------------
# ROUTER NODE
# -------------------------
def route_node(state: AgentState):

    print("👉 router_node")

    prompt = f"""
        Extract structured info from input.

        Return ONLY valid JSON like this:

        {{
        "type": "patient" OR "text",
        "patient_id": "string or null"
        }}

        Rules:
        - If user refers to a patient → type = "patient"
        - Extract exact patient id if possible
        - If no clear id → patient_id = null
        - DO NOT explain anything
        - ONLY return JSON

        Input:
        {state["input"]}
        """
    
    # llm = get_llm()
    # res = llm.invoke(prompt).content.strip()

    res = call_llm(prompt)

    try:
        data = json.loads(res)

        route = data.get("type", "text")
        patient_id = data.get("patient_id")

        print("🧠 ROUTE:", route)
        print("🆔 ID:", patient_id)

        return {
            "route": route,
            "patient_id": patient_id
        }

    except Exception as e:
        print("❌ ROUTER ERROR:", e)

        return {
            "route": "text",
            "patient_id": None
        }


# -------------------------
# PATIENT FLOW
# -------------------------
def fetch_node(state: AgentState):

    print("👉 fetch_node")

    patient_id = state.get("patient_id")

    print("🆔 USING ID:", patient_id)

    if not patient_id:
        return {
            **state,
            "features": {}
        }

    result = fetch_patient.invoke({
        "patient_id": patient_id
    })

    return {
        **state,
        "features": result.get("features", {})
    }

# -------------------------
# TEXT FLOW
# -------------------------
def parse_node(state: AgentState):

    print("👉 parse_node")

    parsed = parse_features.invoke({
        "text": state["input"]
    })

    if parsed.get("status") != "ok":
        return {
            "valid": False,
            "response": parsed.get(
                "message",
                "❌ No valid medical information found."
            )
        }

    features = parsed["data"]["features"]

    return {
        "features": features,
        "valid": True
    }


def validate_node(state: AgentState):

    print("👉 validate_node")

    validated = validate_features.invoke({
        "features": state["features"]
    })

    return {"features": validated["data"]["features"]}


def complete_node(state: AgentState):

    print("👉 complete_node")

    completed = complete_features.invoke({
        "features": state["features"]
    })

    if completed.get("status") != "ok":
        return {
            "valid": False,
            "response": completed.get("message", "Completion failed")
        }

    return {
        "features": completed["data"]["features"],
        "valid": True
    }

# -------------------------
# PREDICT
# -------------------------
def predict_node(state):

    print("👉 predict_node")

    features = state.get("features", {})

    pred = predict.invoke({
        "features": features
    })

    return {"prediction": pred}


# -------------------------
# RESPONSE
# -------------------------
def response_node(state):

    print("👉 response_node")

    if not state.get("valid", True):
        return {
            "response": state.get("response", "❌ Invalid input")
        }

    pred = state.get("prediction", {})

    if pred.get("status") != "ok":
        return {
            "response": f"Error: {pred.get('message')}"
        }

    risk = pred.get("risk")
    prob = pred.get("probability")
    analysis = pred.get("analysis", "")
    top_features = pred.get("top_features", [])

    features_text = "\n".join([
        f"- {f['feature']} = {f['value']} (contribution={f['contribution']:.3f})"
        for f in top_features[:5]
    ])

    return {
        "response": f"""
            Risk: {risk}
            Probability: {prob:.3f}

            Top Features:
            {features_text if features_text else "No important features"}

            AI Analysis:
            {analysis}
            """
    }

# -------------------------
# GRAPH
# -------------------------
def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)

    graph.add_node("fetch", fetch_node)

    graph.add_node("parse", parse_node)
    graph.add_node("validate", validate_node)
    graph.add_node("complete", complete_node)

    graph.add_node("predict", predict_node)
    graph.add_node("respond", response_node)

    graph.set_entry_point("route")

    # routing
    graph.add_conditional_edges(
        "route",
        lambda x: x["route"],
        {
            "patient": "fetch",
            "text": "parse"
        }
    )

    # patient flow
    graph.add_edge("fetch", "predict")

    # text flow
    graph.add_conditional_edges(
            "parse",
            lambda x: x.get("valid", True),
            {
                True: "validate",
                False: "respond"
            }
        )
    graph.add_edge("validate", "complete")

    graph.add_conditional_edges(
        "complete",
        lambda x: x.get("valid", True),
        {
            True: "predict",
            False: "respond"
        }
    )
    
    # final
    graph.add_edge("predict", "respond")
    graph.add_edge("respond", END)

    # compile graph
    graph = graph.compile()

    # draw graph
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("langgraph.png", "wb") as f:
        f.write(png_bytes)

    return graph