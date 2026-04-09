import streamlit as st
from agent.agent import Agent

agent = Agent()

st.title("Lung Cancer Agent")

user_input = st.text_input("Ask something")

if st.button("Run"):
    features = {
        "age": 65,
        "tumor_size": 3.2
    }

    result = agent.run(user_input, features)
    st.write(result)