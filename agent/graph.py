from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List, Optional
import json
import re

from agent.llm import call_llm
from agent.memory import get_checkpointer
from agent.clinical_memory_service import ClinicalMemoryService
from tools.provider_factory import get_tool_provider


memory_service = ClinicalMemoryService()
tool_provider = get_tool_provider()


class AgentState(TypedDict, total=False):
    input: str
    route: str
    features: Dict[str, Any]
    patient_id: Optional[str]
    prediction: Dict[str, Any]
    response: str
    valid: bool
    actor_id: str
    session_id: str
    clinician_preferences: Dict[str, Any]
    session_summary_texts: List[str]
    planner_decision: Dict[str, Any]
    session_preferences_update: Dict[str, Any]
    followup_intent: str


def safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {}


def normalize_features_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    data = result.get("data", {})
    if isinstance(data, dict) and isinstance(data.get("features"), dict):
        return data["features"]

    features = result.get("features")
    if isinstance(features, dict):
        return features

    return {}


def merge_preferences(
    stored_prefs: Optional[Dict[str, Any]],
    planner_update: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged = {}
    if isinstance(stored_prefs, dict):
        merged.update(stored_prefs)
    if isinstance(planner_update, dict):
        merged.update(planner_update)
    return merged


PATIENT_ID_RE = re.compile(r"^[A-Za-z]\d{2}-\d{3}$|^R\d{2}-\d{3}$")


def looks_like_patient_id(text: str) -> bool:
    if not text:
        return False

    return bool(PATIENT_ID_RE.match(text.strip()))


def llm_react_plan(
    user_input: str,
    current_patient_id: Optional[str],
    session_summary_texts: Optional[List[str]],
    clinician_preferences: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    prompt = f"""
        You are a clinical workflow planner for an NSCLC survival prediction assistant.

        Your job:
        - Decide whether the user wants:
        1) patient lookup / patient-based prediction flow
        2) free-text clinical feature parsing flow
        - Resolve ambiguous follow-up questions using memory context
        - Extract preference changes such as concise/detailed, show top features, top 3/top 5, technical/plain wording

        IMPORTANT:
        - Think step by step internally, but DO NOT reveal your reasoning.
        - Return ONLY valid JSON.
        - If the user is referring to the previous patient/case in the current session, set "use_last_case": true.
        - If a clear patient id is present, extract it.
        - If no patient id is present but this is clearly a follow-up to the current session context, use action="patient" and use_last_case=true.
        - If the user is giving fresh raw medical description, use action="text".
        - This system predicts survival risk, not diagnosis.

        Memory context:
        current_patient_id = {json.dumps(current_patient_id)}
        session_summary_texts = {json.dumps(session_summary_texts or [], ensure_ascii=False)}
        clinician_preferences = {json.dumps(clinician_preferences or {}, ensure_ascii=False)}

        Return ONLY JSON with this exact schema:
        {{
        "action": "patient" | "text",
        "patient_id": "string or null",
        "use_last_case": true | false,
        "followup_intent": "none" | "explain" | "rerun_prediction" | "top_features" | "confidence" | "summary",
        "preferences_update": {{
            "response_style": "concise" | "detailed" | null,
            "tone": "technical" | "plain" | null,
            "show_top_features": true | false | null,
            "top_k_features": 3 | 5 | null
        }}
        }}

        User input:
        {user_input}
        """

    raw = call_llm(prompt)
    data = safe_json_loads(raw)

    if not data:
        return {
            "action": "text",
            "patient_id": None,
            "use_last_case": False,
            "followup_intent": "none",
            "preferences_update": {}
        }

    prefs = data.get("preferences_update", {})
    if not isinstance(prefs, dict):
        prefs = {}

    prefs = {k: v for k, v in prefs.items() if v is not None}

    action = data.get("action", "text")

    # enforce schema
    if action not in ["patient", "text"]:
        if action in ["explain", "summary", "top_features", "confidence", "rerun_prediction"]:
            action = "patient"
        else:
            action = "text"

    return {
        "action": action,
        "patient_id": data.get("patient_id"),
        "use_last_case": bool(data.get("use_last_case", False)),
        "followup_intent": data.get("followup_intent", "none"),
        "preferences_update": prefs
    }


def load_memory_context_node(state: AgentState):
    print("👉 load_memory_context_node")

    user_input = state.get("input", "")

    # Fast path: if user entered patient ID, skip AgentCore Memory read
    if looks_like_patient_id(user_input):
        return {
            "clinician_preferences": {},
            "session_summary_texts": []
        }

    actor_id = state.get("actor_id", "default-user")
    session_id = state.get("session_id", "default-session")

    preferences_result = memory_service.get_preferences(
        actor_id=actor_id,
        query=user_input or "clinician output preferences"
    )

    summary_result = memory_service.get_session_summary(
        actor_id=actor_id,
        session_id=session_id,
        query=user_input or "summary of this session"
    )

    return {
        "clinician_preferences": preferences_result.get("parsed", {}),
        "session_summary_texts": summary_result.get("raw_texts", [])
    }


def route_node(state: AgentState):
    print("👉 route_node (LLM planner)")

    current_patient_id = state.get("patient_id")
    session_summary_texts = state.get("session_summary_texts", [])
    clinician_preferences = state.get("clinician_preferences", {})
    user_input = state["input"]

    # Fast path: direct patient ID should not go to LLM planner
    if looks_like_patient_id(user_input):

        patient_id = user_input.strip()

        return {
            "route": "patient",
            "patient_id": patient_id,
            "planner_decision": {},
            "session_preferences_update": {},
            "followup_intent": "none"
        }

    plan = llm_react_plan(
        user_input=user_input,
        current_patient_id=current_patient_id,
        session_summary_texts=session_summary_texts,
        clinician_preferences=clinician_preferences
    )

    raw_action = plan.get("action", "text")
    patient_id = plan.get("patient_id")
    use_last_case = plan.get("use_last_case", False)
    followup_intent = plan.get("followup_intent", "none")
    preferences_update = plan.get("preferences_update", {})

    # NORMALIZE ACTION
    if raw_action in ["explain", "summary", "top_features", "confidence", "rerun_prediction"]:
        route = "patient" if use_last_case or current_patient_id else "text"
    elif raw_action in ["patient", "text"]:
        route = raw_action
    else:
        route = "text"

    # REUSE CURRENT SESSION PATIENT
    if use_last_case and not patient_id:
        patient_id = current_patient_id


    if route == "patient" and not patient_id:
        return {
            "route": "text",
            "patient_id": None,
            "planner_decision": plan,
            "session_preferences_update": preferences_update,
            "followup_intent": followup_intent,
            "valid": False,
            "response": "❌ No previous patient context found in this session."
        }

    print("PLAN:", plan)
    print("CURRENT PATIENT ID:", current_patient_id)
    print("FINAL PATIENT ID:", patient_id)
    print("ROUTE:", route)

    return {
        "route": route,
        "patient_id": patient_id,
        "planner_decision": plan,
        "session_preferences_update": preferences_update,
        "followup_intent": followup_intent
    }


def fetch_node(state: AgentState):
    print("👉 fetch_node")

    patient_id = state.get("patient_id")

    print("FETCH PATIENT ID:", patient_id)

    if not patient_id:
        return {
            "features": {},
            "valid": False,
            "response": "❌ No patient_id available."
        }

    result = tool_provider.fetch_patient(patient_id)

    if result.get("status") != "ok":
        return {
            "features": {},
            "valid": False,
            "response": result.get(
                "message",
                f"❌ Failed to fetch features for patient {patient_id}"
            )
        }

    features = normalize_features_result(result)

    if not features:
        return {
            "features": {},
            "valid": False,
            "response": f"❌ No usable features found for patient {patient_id}"
        }

    return {
        "features": features,
        "valid": True,
        "patient_id": patient_id
    }


def parse_node(state: AgentState):
    print("👉 parse_node")

    parsed = tool_provider.parse_features(state["input"])

    if parsed.get("status") != "ok":
        return {
            "valid": False,
            "response": parsed.get(
                "message",
                "❌ No valid medical information found."
            )
        }

    features = normalize_features_result(parsed)

    if not features:
        return {
            "valid": False,
            "response": "❌ Parsed output did not contain usable features."
        }

    return {
        "features": features,
        "valid": True,
        "patient_id": state.get("patient_id")

    }


def validate_node(state: AgentState):
    print("👉 validate_node")

    validated = tool_provider.validate_features(state["features"])

    if validated.get("status") != "ok":
        return {
            "valid": False,
            "response": validated.get("message", "❌ Validation failed")
        }

    features = normalize_features_result(validated)

    if not features:
        return {
            "valid": False,
            "response": "❌ Validation output did not contain usable features."
        }

    return {
        "features": features,
        "valid": True
    }


def complete_node(state: AgentState):
    print("👉 complete_node")

    completed = tool_provider.complete_features(state["features"])

    if completed.get("status") != "ok":
        return {
            "valid": False,
            "response": completed.get("message", "Completion failed")
        }

    features = normalize_features_result(completed)

    if not features:
        return {
            "valid": False,
            "response": "❌ Completion output did not contain usable features."
        }

    return {
        "features": features,
        "valid": True
    }


def predict_node(state: AgentState):
    print("👉 predict_node")

    features = state.get("features", {})
    pred = tool_provider.predict(features)

    return {
        "prediction": pred
    }


def response_node(state: AgentState):
    print("👉 response_node")

    if not state.get("valid", True):
        return {
            "response": state.get("response", "❌ Invalid input")
        }

    pred = state.get("prediction", {})

    print("PREDICTION:", pred)
    print("FOLLOWUP_INTENT:", state.get("followup_intent"))

    if pred.get("status") != "ok":
        return {
            "response": f"Error: {pred.get('message')}"
        }

    stored_prefs = state.get("clinician_preferences", {})
    planner_update = state.get("session_preferences_update", {})
    prefs = merge_preferences(stored_prefs, planner_update)



    risk = pred.get("risk")
    prob = pred.get("probability")
    analysis = pred.get("analysis", "")
    top_features = pred.get("top_features", [])

    response_style = prefs.get("response_style", "detailed")
    show_top_features = prefs.get("show_top_features", True)
    top_k_features = int(prefs.get("top_k_features", 5))

    features_text = "\n".join([
        f"- {f['feature']} = {f['value']} (contribution={f['contribution']:.3f})"
        for f in top_features[:top_k_features]
    ])

    if response_style == "concise":
        parts = [
            f"Risk: {risk}",
            f"Probability: {prob:.3f}",
        ]

        if show_top_features and features_text:
            parts.append(f"Top Features:\n{features_text}")

        if analysis:
            parts.append(f"AI Analysis:\n{analysis}")

        response = "\n\n".join(parts)

    else:
        parts = [
            f"Risk: {risk}",
            f"Probability: {prob:.3f}",
        ]

        if show_top_features:
            parts.append(
                f"Top Features:\n{features_text if features_text else 'No important features'}"
            )

        if analysis:
            parts.append(f"AI Analysis:\n{analysis}")

        response = "\n\n".join(parts)

    return {
        "response": response.strip(),
        "clinician_preferences": prefs
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_memory_context", load_memory_context_node)
    graph.add_node("route", route_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("parse", parse_node)
    graph.add_node("validate", validate_node)
    graph.add_node("complete", complete_node)
    graph.add_node("predict", predict_node)
    graph.add_node("respond", response_node)

    graph.set_entry_point("load_memory_context")

    graph.add_edge("load_memory_context", "route")

    graph.add_conditional_edges(
        "route",
        lambda x: x["route"],
        {
            "patient": "fetch",
            "text": "parse"
        }
    )

    graph.add_edge("fetch", "predict")

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

    graph.add_edge("predict", "respond")
    graph.add_edge("respond", END)

    checkpointer = get_checkpointer()
    graph = graph.compile(checkpointer=checkpointer)

    # png_bytes = graph.get_graph().draw_mermaid_png()
    # with open("langgraph.png", "wb") as f:
    #     f.write(png_bytes)

    return graph