import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from LangGraph_graph.Graph import graph  # This builds and compiles your LangGraph app

# === Session Initialization ===
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "new_user-123"  # Can be dynamic if needed

if "messages" not in st.session_state:
    st.session_state.messages = []  # For chat history

if "interrupted" not in st.session_state:
    st.session_state.interrupted = False  # Flag for HITL interruption

if "interrupted_output" not in st.session_state:
    st.session_state.interrupted_output = None  # Store interrupted response content


# === Load LangGraph App ===
# @st.cache_resource
def get_graph():
    return graph()  # This should return a compiled LangGraph

app = graph()


# === App Title ===
st.title("🧠 Multi-Agent Assistant with HITL")

# === Display Chat History ===
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# === Chat Input ===
if prompt := st.chat_input("Ask a question or request a post..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save to history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Build input state
    state = {
        "query": prompt,
        "messages": st.session_state.messages,
        "intermediate_data":{},
        "next_node": None,
        "response": None,
    }

    # === Run the LangGraph ===
    with st.spinner("Thinking..."):
        result = app.invoke(
            state,
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

    # === If interrupted (HITL) ===
    if "__type__" in result and result["__type__"] == "human":
        st.session_state.interrupted = True
        st.session_state.interrupted_output = result["content"]

    # === Normal result ===
    else:
        response = result.get("response", "(No response)")
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))


# === Handle HITL Feedback ===
if st.session_state.interrupted:
    st.divider()
    st.subheader("🛑 Human-in-the-Loop Review")
    st.markdown("🧾 Agent Output Pending Review:")
    st.info(st.session_state.interrupted_output)

    feedback = st.text_area("✍️ Suggest edits or approve as-is:", key="feedback_input")

    if st.button("Submit Feedback"):
        with st.spinner("Submitting feedback and resuming..."):
            resumed = app.resume(
                HumanMessage(content=feedback),
                thread_id=st.session_state.thread_id
            )

        # Display resumed result
        response = resumed.get("response", "(No response)")
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))

        # Reset HITL state
        st.session_state.interrupted = False
        st.session_state.interrupted_output = None
