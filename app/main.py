import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from LangGraph_graph.Graph import graph  # This builds and compiles your LangGraph app

# === Session Initialization ===
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-123"  # Can be dynamic later
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history

# === Get LangGraph App ===
@st.cache_resource
def get_graph():
    return graph()

app = get_graph()

# === App Title ===
st.title("🧠 Multi-Agent Assistant")

# === Display Chat History ===
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# === Input Field ===
if prompt := st.chat_input("Ask a question or request a post..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Prepare state for LangGraph
    state = {
        "query": prompt,
        "messages": st.session_state.messages,
        "next_node": None,
        "response": None,
    }

    with st.spinner("Thinking..."):
    # Call the graph
        result = app.invoke(
            state,
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

        # Get and show response
        response = result.get("response", "(No response)")

    with st.chat_message("assistant"):
        st.markdown(response)

    # Save response to history
    st.session_state.messages.append(AIMessage(content=response))
