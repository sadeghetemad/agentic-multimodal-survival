import os
import uuid
from urllib.parse import quote_plus

import requests
import streamlit as st


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="NSCLC Survival Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------
# CONSTANTS
# -------------------------
OHIF_BASE_URL = "https://dxhhnitg56xjv.cloudfront.net/"

AGENTCORE_URL = os.getenv(
    "AGENTCORE_URL",
    "http://localhost:8080"
)


# -------------------------
# SESSION STATE
# -------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "System"

if "actor_id" not in st.session_state:
    st.session_state.actor_id = "clinician_demo_1"

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None


# -------------------------
# HELPERS
# -------------------------
def create_new_session():
    new_session_id = str(uuid.uuid4())
    st.session_state.sessions[new_session_id] = {
        "title": "New chat",
        "messages": []
    }
    st.session_state.current_session_id = new_session_id


def get_current_session():
    if st.session_state.current_session_id is None:
        create_new_session()

    return st.session_state.sessions[st.session_state.current_session_id]


def build_ohif_patient_url(patient_id: str):
    return f"{OHIF_BASE_URL}?mrn={quote_plus(patient_id)}"


def call_agentcore(input_text: str, session_id: str, actor_id: str):
    payload = {
        "input": input_text,
        "session_id": session_id,
        "actor_id": actor_id
    }

    try:
        response = requests.post(
            AGENTCORE_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "AgentCore request timed out."
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": (
                "Could not connect to AgentCore backend. "
                "Make sure the backend is running."
            )
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"AgentCore request failed: {str(e)}"
        }


def extract_probability(output: str):
    import re

    prob_match = re.search(r"Probability:\s*([0-9.]+)", output)
    if prob_match:
        try:
            return float(prob_match.group(1))
        except Exception:
            return None
    return None


def extract_analysis(output: str):
    if "AI Analysis:" in output:
        analysis = output.split("AI Analysis:")[-1].strip()
    else:
        cleaned_lines = []

        for line in output.splitlines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("-") and "contribution=" in line:
                continue

            if line.lower().startswith("probability:"):
                continue

            cleaned_lines.append(line)

        analysis = "\n".join(cleaned_lines).strip()

    filtered_lines = []

    unwanted_prefixes = [
        "# nsclc survival risk assessment",
        "risk classification:",
        "mortality probability:"
    ]

    for line in analysis.splitlines():
        stripped = line.strip()

        if any(stripped.lower().startswith(p) for p in unwanted_prefixes):
            continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


# -------------------------
# SIDEBAR TOP
# -------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🫁</div>
            <div>
                <div class="sidebar-title">NSCLC AI</div>
                <div class="sidebar-subtitle">Clinical assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section-title">Theme</div>',
        unsafe_allow_html=True
    )

    st.session_state.theme_mode = st.radio(
        "Theme",
        ["System", "Light", "Dark"],
        horizontal=False,
        label_visibility="collapsed"
    )


# -------------------------
# THEME CSS
# -------------------------
theme_mode = st.session_state.theme_mode

if theme_mode == "Light":
    css_theme = """
    :root {
        --bg: #f7f9fc;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --text: #101828;
        --muted: #667085;
        --border: #e4e7ec;
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --primary-soft: #eff6ff;
        --success-bg: #ecfdf3;
        --success-text: #027a48;
        --danger-bg: #fef3f2;
        --danger-text: #b42318;
        --shadow: 0 10px 28px rgba(16,24,40,0.06);
    }
    """
elif theme_mode == "Dark":
    css_theme = """
    :root {
        --bg: #080d19;
        --surface: #101827;
        --surface-soft: #121d31;
        --text: #f8fafc;
        --muted: #98a2b3;
        --border: #263348;
        --primary: #3b82f6;
        --primary-hover: #60a5fa;
        --primary-soft: #13284a;
        --success-bg: #12372b;
        --success-text: #86efac;
        --danger-bg: #3b1717;
        --danger-text: #fca5a5;
        --shadow: 0 10px 28px rgba(0,0,0,0.28);
    }
    """
else:
    css_theme = """
    :root {
        --bg: #f7f9fc;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --text: #101828;
        --muted: #667085;
        --border: #e4e7ec;
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --primary-soft: #eff6ff;
        --success-bg: #ecfdf3;
        --success-text: #027a48;
        --danger-bg: #fef3f2;
        --danger-text: #b42318;
        --shadow: 0 10px 28px rgba(16,24,40,0.06);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #080d19;
            --surface: #101827;
            --surface-soft: #121d31;
            --text: #f8fafc;
            --muted: #98a2b3;
            --border: #263348;
            --primary: #3b82f6;
            --primary-hover: #60a5fa;
            --primary-soft: #13284a;
            --success-bg: #12372b;
            --success-text: #86efac;
            --danger-bg: #3b1717;
            --danger-text: #fca5a5;
            --shadow: 0 10px 28px rgba(0,0,0,0.28);
        }
    }
    """

st.markdown(f"""
<style>
{css_theme}

.stApp {{
    background: var(--bg);
    color: var(--text);
}}

.block-container {{
    max-width: 1160px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}}

section[data-testid="stSidebar"] {{
    width: 218px !important;
    min-width: 218px !important;
    border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] > div {{
    background: var(--surface);
    padding: 0.75rem 0.75rem 1rem 0.75rem;
}}

section[data-testid="stSidebar"] * {{
    font-size: 12px !important;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 16px;
}}

.sidebar-logo {{
    width: 30px;
    height: 30px;
    border-radius: 10px;
    background: var(--primary-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px !important;
}}

.sidebar-title {{
    font-size: 13px !important;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
}}

.sidebar-subtitle {{
    font-size: 10.5px !important;
    color: var(--muted);
    margin-top: 2px;
}}

.sidebar-section-title {{
    font-size: 11px !important;
    font-weight: 800;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 12px 0 6px 0;
}}

section[data-testid="stSidebar"] .stRadio label {{
    font-size: 11.5px !important;
    color: var(--text) !important;
}}

h1, h2, h3, h4, p, span, label {{
    color: var(--text);
}}

.header-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 22px;
    box-shadow: var(--shadow);
    margin-bottom: 26px;
}}

.header-title {{
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 0;
}}

.header-subtitle {{
    color: var(--muted);
    margin-top: 4px;
    font-size: 13px;
}}

.assessment-header {{
    margin-bottom: 14px;
}}

.assessment-title-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}}

.assessment-icon {{
    width: 34px;
    height: 34px;
    border-radius: 12px;
    background: var(--primary-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
}}

.assessment-title {{
    font-size: 22px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
}}

.assessment-helper {{
    color: var(--muted);
    font-size: 13px;
    margin-left: 44px;
}}

.user-label {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 3px;
}}

.user-query {{
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 14px;
}}

.ohif-card {{
    background: var(--primary-soft);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 13px;
    margin: 10px 0 15px 0;
}}

.ohif-title {{
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 4px;
}}

.ohif-meta {{
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 10px;
}}

.ohif-button {{
    display: inline-block;
    padding: 7px 11px;
    border-radius: 9px;
    background: var(--primary);
    color: white !important;
    text-decoration: none;
    font-size: 12px;
    font-weight: 800;
}}

.ohif-button:hover {{
    background: var(--primary-hover);
}}

.risk-box {{
    border-radius: 15px;
    padding: 15px;
    min-height: 88px;
    border: 1px solid var(--border);
}}

.risk-low {{
    background: var(--success-bg);
    color: var(--success-text);
}}

.risk-high {{
    background: var(--danger-bg);
    color: var(--danger-text);
}}

.risk-title {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 7px;
}}

.risk-label {{
    font-size: 18px;
    font-weight: 800;
}}

.risk-value {{
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.03em;
}}

.insight-card {{
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 14px;
    margin-top: 15px;
    line-height: 1.65;
    font-size: 14px;
}}

.stTextInput > div > div > input {{
    height: 38px;
    border-radius: 10px;
    background: var(--surface-soft);
    color: var(--text);
    border: 1px solid var(--border);
    font-size: 13px;
}}

.stTextInput > div > div > input:focus {{
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
}}

.stButton > button {{
    height: 38px;
    border-radius: 10px;
    background: var(--primary);
    color: white;
    border: none;
    font-size: 13px;
    font-weight: 800;
    box-shadow: 0 6px 14px rgba(37,99,235,0.22);
    transition: all 0.16s ease-in-out;
}}

.stButton > button:hover {{
    background: var(--primary-hover);
    color: white;
    border: none;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(37,99,235,0.28);
}}

.stButton > button:active {{
    transform: translateY(0);
    box-shadow: 0 4px 10px rgba(37,99,235,0.20);
}}

section[data-testid="stSidebar"] .stButton > button {{
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 9px !important;
    border-radius: 9px !important;
    background: var(--surface-soft) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}}

section[data-testid="stSidebar"] .stButton > button:hover {{
    background: var(--primary-soft) !important;
    border-color: var(--primary) !important;
    color: var(--text) !important;
    transform: translateY(-1px);
}}

.sidebar-chat-label {{
    font-size: 10.5px !important;
    color: var(--muted);
    margin: 4px 0 6px 0;
}}

div[data-testid="stProgress"] > div > div > div {{
    background-color: var(--primary);
}}

hr {{
    border-color: var(--border);
}}
</style>
""", unsafe_allow_html=True)


# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">🫁 NSCLC Survival Predictor</div>
    <div class="header-subtitle">
        AI-powered survival risk assessment with imaging access
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------
# INPUT
# -------------------------
st.markdown("""
<div class="assessment-header">
    <div class="assessment-title-row">
        <div class="assessment-icon">🩺</div>
        <div class="assessment-title">Patient Risk Assessment</div>
    </div>
    <div class="assessment-helper">
        Enter the ID of the patient you want to assess.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.text_input(
        "Enter patient ID",
        placeholder="e.g., R01-029",
        label_visibility="collapsed"
    )

with col2:
    run_button = st.button("Run", use_container_width=True)


# -------------------------
# RUN AGENT
# -------------------------
if run_button and user_input:
    with st.spinner("Processing clinical query..."):
        try:
            current_session = get_current_session()
            current_session_id = st.session_state.current_session_id

            response = call_agentcore(
                input_text=user_input,
                session_id=current_session_id,
                actor_id=st.session_state.actor_id
            )

            status = response.get("status", "error")

            if status != "ok":
                st.error(response.get("message", "Unknown error"))
            else:
                result = response.get("response", "")
                patient_id = response.get("patient_id")

                ohif_url = (
                    build_ohif_patient_url(patient_id)
                    if patient_id
                    else None
                )

                current_session["messages"].append({
                    "input": user_input,
                    "output": result,
                    "session_id": current_session_id,
                    "actor_id": st.session_state.actor_id,
                    "patient_id": patient_id,
                    "ohif_url": ohif_url,
                })

                if current_session["title"] == "New chat":
                    current_session["title"] = user_input[:28] + (
                        "..." if len(user_input) > 28 else ""
                    )

                st.toast("Prediction ready")

        except Exception as e:
            st.error(str(e))


# -------------------------
# OUTPUT
# -------------------------
messages = []

if st.session_state.current_session_id is not None:
    current_session = st.session_state.sessions.get(
        st.session_state.current_session_id
    )
    if current_session:
        messages = current_session.get("messages", [])


if messages:
    st.markdown("### Results")

    for item in reversed(messages):
        output = item["output"]

        if not output:
            st.warning("I couldn't understand your request.")
            continue

        if "not found" in output.lower():
            st.error("Patient not found")
            continue

        prob = extract_probability(output)
        analysis = extract_analysis(output)

        patient_id = item.get("patient_id")
        ohif_url = item.get("ohif_url")

        with st.container():
            st.markdown(
                f"""
                <div class="user-label">You searched</div>
                <div class="user-query">{item["input"]}</div>
                """,
                unsafe_allow_html=True
            )

            if patient_id and ohif_url:
                st.markdown(f"""
                <div class="ohif-card">
                    <div class="ohif-title">Medical Imaging</div>
                    <div class="ohif-meta">Patient ID: <b>{patient_id}</b></div>
                    <a class="ohif-button" href="{ohif_url}" target="_blank">
                        Open in OHIF
                    </a>
                </div>
                """, unsafe_allow_html=True)

            if prob is not None:
                risk_label = "High Risk" if prob >= 0.5 else "Low Risk"
                risk_class = "risk-high" if prob >= 0.5 else "risk-low"

                col_a, col_b = st.columns([1, 2])

                with col_a:
                    st.markdown(
                        f"""
                        <div class="risk-box {risk_class}">
                            <div class="risk-title">Risk Category</div>
                            <div class="risk-label">{risk_label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_b:
                    st.markdown(
                        f"""
                        <div class="risk-box">
                            <div class="risk-title">Mortality Risk</div>
                            <div class="risk-value">{prob:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(prob)

            if analysis:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div>{analysis}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# -------------------------
# SIDEBAR CHAT CONTROLS
# -------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-section-title">Chats</div>',
        unsafe_allow_html=True
    )

    if st.button("＋ New Chat", use_container_width=True):
        create_new_session()
        st.rerun()

    session_items = list(st.session_state.sessions.items())
    session_items.reverse()

    visible_sessions = [
        (sess_id, sess_data)
        for sess_id, sess_data in session_items
        if sess_data.get("messages")
    ]

    if visible_sessions:
        st.markdown(
            '<div class="sidebar-chat-label">Recent</div>',
            unsafe_allow_html=True
        )

    for sess_id, sess_data in visible_sessions:
        title = sess_data.get("title", "New chat")

        if st.button(
            f"💬 {title}",
            key=f"session_btn_{sess_id}",
            use_container_width=True
        ):
            st.session_state.current_session_id = sess_id
            st.rerun()

    current_session = None
    if st.session_state.current_session_id is not None:
        current_session = st.session_state.sessions.get(
            st.session_state.current_session_id
        )

    if current_session and current_session.get("messages"):
        st.markdown("---")
        st.markdown(
            '<div class="sidebar-section-title">Controls</div>',
            unsafe_allow_html=True
        )

        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.sessions[
                st.session_state.current_session_id
            ]["messages"] = []

            st.session_state.sessions[
                st.session_state.current_session_id
            ]["title"] = "New chat"

            st.rerun()

        if st.button("🗑️ Delete", use_container_width=True):
            current_id = st.session_state.current_session_id

            if current_id in st.session_state.sessions:
                del st.session_state.sessions[current_id]

            if st.session_state.sessions:
                st.session_state.current_session_id = list(
                    st.session_state.sessions.keys()
                )[0]
            else:
                st.session_state.current_session_id = None

            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-section-title">Debug</div>',
        unsafe_allow_html=True
    )

    st.caption(f"actor: `{st.session_state.actor_id}`")

    if st.session_state.current_session_id:
        st.caption(
            f"session: `{st.session_state.current_session_id[:8]}...`"
        )
    else:
        st.caption("session: `not started`")

    st.caption(f"backend: `{AGENTCORE_URL}`")

    st.markdown("---")
    st.caption("Powered by Amazon SageMaker, Bedrock, AgentCore and LangGraph")