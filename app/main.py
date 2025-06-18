# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from LangGraph_graph.Graph import GenAIMultiAgentRunner # Your compiled LangGraph app
from utils.streamlit_utils import (
    initialize_session,
    display_chat_history,
    handle_paused_state,
    handle_chat_input
)

# === UI Header ===
st.set_page_config(page_title="Multi-Agent Assistant", page_icon="🧠")
st.title("🧠 Multi-Agent Assistant")
st.markdown(
    """
    Welcome to the Multi-Agent Assistant. This system uses a graph-based multi-agent setup. 
    """
)

# === Initialize Session State ===
initialize_session()

# === Load Graph App ===
@st.cache_resource
def get_graph():
    return GenAIMultiAgentRunner()

app = get_graph().get_app()

# === Display Chat History ===
display_chat_history(st.session_state.messages)

# === Handle HITL Pause State ===
if st.session_state.paused_state:
    handle_paused_state(app)

# === Handle Standard Chat Input ===
else:
    handle_chat_input(app)
