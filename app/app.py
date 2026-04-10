import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from agent.agent import Agent
import pandas as pd

# -------------------------
# INIT
# -------------------------

st.set_page_config(
    page_title="Medical AI Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Medical NSCLC Survival Prediction Agent")

st.markdown("Enter a patient ID or custom query to get prediction and analysis.")

# -------------------------
# SESSION STATE
# -------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

agent = st.session_state.agent


# -------------------------
# INPUT UI
# -------------------------

with st.container():

    user_input = st.text_input(
        "Enter your request",
        placeholder="e.g. Predict survival for patient R01-029"
    )

    run_button = st.button("🚀 Run Prediction")


# -------------------------
# RUN AGENT
# -------------------------

if run_button and user_input:

    with st.spinner("Running AI Agent..."):

        try:
            result = agent.run(user_input, {})

            # store history
            st.session_state.history.append({
                "input": user_input,
                "output": result
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


# -------------------------
# OUTPUT UI
# -------------------------

# -------------------------
# OUTPUT UI (BEAUTIFIED)
# -------------------------

st.divider()
st.subheader("📊 Results")

if st.session_state.history:

    for item in reversed(st.session_state.history):

        st.markdown("### 🧑‍💻 User Query")
        st.code(item["input"], language="text")

        output = item["output"]

        # -------------------------
        # PARSE OUTPUT (hacky but works)
        # -------------------------
        st.markdown("### 🤖 Prediction")

        # رنگ برای risk
        if "high" in output.lower():
            st.error("🔴 High Risk")
        elif "low" in output.lower():
            st.success("🟢 Low Risk")
        else:
            st.warning("⚠️ Unknown Risk")

        # probability extract
        import re
        prob_match = re.search(r"Probability:\s*([0-9.]+)", output)

        if prob_match:
            prob = float(prob_match.group(1))
            st.progress(prob)
            st.caption(f"Probability: {prob:.3f}")

        st.markdown("---")

        # -------------------------
        # FEATURES
        # -------------------------
        if "Top Contributing Features" in output:

            st.markdown("### 🧬 Key Features")

            lines = output.split("\n")
            features = [l for l in lines if "-" in l and "importance" in l]

            if features:
                feature_data = []

                for f in features:
                    try:
                        name = f.split("=")[0].replace("-", "").strip()
                        value = float(f.split("=")[1].split("(")[0])
                        imp = float(f.split("importance=")[1].replace(")", ""))

                        feature_data.append({
                            "Feature": name,
                            "Value": value,
                            "Importance": imp
                        })
                    except:
                        continue

                if feature_data:
                    st.dataframe(
                        pd.DataFrame(feature_data).sort_values("Importance", ascending=False),
                        use_container_width=True
                    )

        # -------------------------
        # ANALYSIS
        # -------------------------
        if "AI Analysis" in output:

            st.markdown("### 🧠 AI Analysis")

            analysis = output.split("AI Analysis:")[-1]
            st.info(analysis.strip())

        st.divider()

else:
    st.info("No results yet. Try running a prediction.")

# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.header("⚙️ Controls")

    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.success("History cleared")

    st.markdown("---")

    st.markdown("### 💡 Example Queries")

    st.markdown("""
    - Predict survival for patient R01-029  
    - Predict survival for patient R01-055  
    - Analyze patient R01-117  
    """)

    st.markdown("---")
    st.caption("Built with LangGraph + SageMaker + LLM")