import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents')))
from router_agent import router_agent_executer
import streamlit as st
from loguru import logger

# Store logs in a file with rotation and retention settings
logger.add("logs/app.log")

st.set_page_config(page_title="GenAI Multi-Agent UI", page_icon="🧠")
st.title("🧠 GenAI Multi-Agent System")


user_query = st.text_area("Enter your query:")

if st.button("Submit"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing your query..."):
            result = router_agent_executer(user_query)  
        st.markdown("### ✅ Final Output")
        st.write(result)
