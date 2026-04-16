import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import uuid
import re

from app.agentcore_app import handler


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="NSCLC Survival Predictor",
    page_icon="🫁",
    layout="wide"
)


# -------------------------
# STYLE
# -------------------------
st.markdown("""
        <style>
        .main {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
        }

        .block-container {
            padding-top: 2rem;
        }

        .header-card {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }

        .stTextInput>div>div>input {
            border-radius: 12px;
            height: 48px;
            padding: 10px;
        }

        .stButton>button {
            background: linear-gradient(135deg, #4A90E2, #6FA8FF);
            color: white;
            border-radius: 12px;
            height: 48px;
            width: 100%;
            font-weight: 600;
            border: none;
        }

        .card {
            background: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }

        .high {background: #ffe5e5; color: #d60000;}
        .low {background: #e6fff2; color: #00994d;}
        </style>
        """, unsafe_allow_html=True)


# -------------------------
# HEADER
# -------------------------
st.markdown("""
    <div class="header-card">
    <h1>🫁 NSCLC Survival Predictor</h1>
    <p style="color:gray;margin-top:-10px;">
    AI-powered clinical decision support system
    </p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------
# SESSION
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# -------------------------
# INPUT
# -------------------------
st.markdown("### 🩺 Patient Risk Assessment")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Enter clinical query",
        placeholder="e.g. Predict survival for patient R01-029",
        label_visibility="collapsed"
    )

with col2:
    run_button = st.button("🚀 Run")


# -------------------------
# RUN AGENT
# -------------------------
if run_button and user_input:
    with st.spinner("🫁 AI Processing..."):
        try:
            response = handler(
                {
                    "input": user_input,
                    "session_id": st.session_state.session_id
                },
                None
            )

            result = response.get("response", "")

            st.session_state.history.append({
                "input": user_input,
                "output": result
            })

            st.toast("Prediction ready")

        except Exception as e:
            st.error(str(e))


# -------------------------
# OUTPUT
# -------------------------
st.markdown("### 💬 Results")

if st.session_state.history:

    for item in reversed(st.session_state.history):

        output = item["output"]

        # User
        st.markdown("**🧑‍💻 You**")
        st.markdown(f"**{item['input']}**")

        # Safety
        if not output:
            st.warning("⚠️ I couldn't understand your request.")
            st.divider()
            continue

        if "not found" in output.lower():
            st.error("🚫 Patient not found")
            st.divider()
            continue

        # Parse probability
        prob = None
        prob_match = re.search(r"Probability:\s*([0-9.]+)", output)
        if prob_match:
            prob = float(prob_match.group(1))

        # Extract features
        features = []
        for f in output.split("\n"):
            if "-" in f and "importance" in f:
                try:
                    name = f.split("=")[0].replace("-", "").strip()
                    imp = float(f.split("importance=")[1].replace(")", ""))
                    features.append((name, imp))
                except:
                    continue

        # Extract analysis
        analysis = ""
        if "AI Analysis" in output:
            analysis = output.split("AI Analysis:")[-1].strip()

        # AI Response
        st.markdown("🤖 **AI Response**")

        col1, col2 = st.columns([1, 2])

        with col1:
            if prob is not None:
                if prob >= 0.5:
                    st.error("🔴 High Risk")
                else:
                    st.success("🟢 Low Risk")

        with col2:
            if prob is not None:
                st.metric("Mortality Risk", f"{prob:.1%}")
                st.progress(prob)

        if analysis:
            st.markdown("🧠 **Insight**")
            st.write(analysis)

        st.divider()

else:
    st.info("Start by entering a patient query.")


# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:

    st.markdown("## 🧹 Controls")

    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared")

    st.markdown("---")
    st.caption("Built with AWS SageMaker + Bedrock + LangGraph")