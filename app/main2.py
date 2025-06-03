# streamlit_app.py

import streamlit as st
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from LangGraph_graph.Graph import graph

# === Initialize or reuse the compiled graph ===
if "langgraph_app" not in st.session_state:
    st.session_state.langgraph_app = graph()

app = st.session_state.langgraph_app

# === Create/retrieve a thread_id to isolate each user session ===
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid4()}"

thread_id = st.session_state.thread_id

# === Streamlit UI ===
st.set_page_config(page_title="GenAI Multi-Agent Chatbot", page_icon="🤖")
st.title("🤖 GenAI Multi-Agent Chatbot")

# === Retrieve full conversation state history (including messages) ===
# app.get_state_history returns a generator of all saved state snapshots for this thread_id.
state_history = list(app.get_state_history(config={"configurable": {"thread_id": thread_id}}))

# If there's at least one saved state, grab the last one to pull the messages array.
if state_history:
    # Each entry is a tuple: (state_dict, metadata)
    last_entry = state_history[-1]
    latest_data = last_entry[0]   # the actual AgentState dict

    for msg in latest_data.get("messages", []):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# === Input box for the user ===
query = st.chat_input("Ask anything…")

if query:
    # Immediately display the user’s message
    with st.chat_message("user"):
        st.markdown(query)

    # Construct a fresh AgentState
    new_state = {
        "query": query,
        "messages": [],             # LangGraph auto‑loads from store internally
        "intermediate_data": {},    # Persisted from previous turns via state
        "next_node": None,
        "None": None
    }

    # Invoke the graph (router + agent) under this thread_id
    result = app.invoke(
        new_state,
        config={"configurable": {"thread_id": thread_id}}
    )

    assistant_reply = result["response"]

    # Display the assistant’s reply
    with st.chat_message("assistant"):      
        st.markdown(assistant_reply)
