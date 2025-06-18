import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage

def initialize_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "test-user-123"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "paused_state" not in st.session_state:
        st.session_state.paused_state = None

def display_chat_history(messages):
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

def handle_paused_state(app):
    st.info("✋ The assistant is waiting for your feedback before continuing.")
    feedback = st.chat_input("Give your feedback to revise or approve the response...")
    if feedback:
        st.session_state.paused_state["query"] = feedback
        with st.spinner("Resuming with feedback..."):
            result = app.resume(
                st.session_state.paused_state,
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
        st.session_state.paused_state = None
        final_response = result.get("response", "(No response)")
        with st.chat_message("assistant"):
            st.markdown(final_response)
        st.session_state.messages.append(AIMessage(content=final_response))

def handle_chat_input(app):
    prompt = st.chat_input("Ask a question or request a post...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))
        state = {
            "query": prompt,
            "messages": st.session_state.messages,
            "intermediate_data":{},
            "next_node": None,
            "response": None,
        }
        with st.spinner("Thinking..."):
            result = app.invoke(
                state,
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
        if isinstance(result, dict) and result.get("__type__") == "interrupt":
            st.session_state.paused_state = result
            st.info("🛑 Human-in-the-loop triggered. Review the content to continue...")
        else:
            response = result.get("response", "(No response)")
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append(AIMessage(content=response))
