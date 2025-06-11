import streamlit as st
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from LangGraph_graph.Graph import graph

# Initialize the LangGraph app
app = graph()

# Streamlit UI setup
st.set_page_config(page_title="GenAI Multi-Agent Chatbot", page_icon="🧠")
st.title("\U0001f9e0 GenAI Multi-Agent Chatbot")

# Generate or retrieve session thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid4()}"

# Retrieve and render chat history using LangGraph state history
state_history = list(app.get_state_history(config={"configurable": {"thread_id": st.session_state.thread_id}}))
if state_history:
    latest_state = state_history[-1]
    for msg in for msg in latest_state.state.get("chat_history", []):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# Handle user input
query = st.chat_input("Ask me anything...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    state = {
        "query": query,
        "intermediate_data": {},
        "response": None,
        "next_node": None,
        "chat_history": []  # LangGraph handles persistence
    }

    with st.spinner("Thinking..."):
        result = app.invoke(state, config={"configurable": {"thread_id": st.session_state.thread_id}})
        response = result["response"]

    with st.chat_message("assistant"):
        st.markdown(response)
