# 🧠 GenAI Multi-Agent Chatbot (with LangGraph)

This project is a fully-functional, modular AI chatbot system built using **LangGraph**. It combines **multi-agent reasoning**, **persistent memory**, and a **Streamlit UI** to create an intelligent assistant that can:

- 🎯 Route queries to the correct agent (router agent)
- 🎥 Generate social media posts from YouTube videos
- 📄 Generate social media posts from web articles
- 📚 Retrieve curated learning resources from YouTube, Coursera, and Udemy
- 🧠 Store conversation history persistently in a SQLite database
- 🔁 Maintain long-term memory across sessions using `thread_id`
- 👁️ Support human-in-the-loop and time-travel (future-ready)

---

## 🗂️ Project Structure

```bash
agents/                      # Individual agents that solve user tasks
├── router_agent.py          # Determines which agent to call
├── topic_learning_agent.py  # Retrieves YouTube/Coursera/Udemy resources
├── youtube_post_agent.py    # Generates posts from YouTube videos
├── article_post_agent.py    # Generates posts from web articles

tool/                        # LangChain-compatible tools
├── youtube_search_tool.py   # Searches YouTube
├── course_search_tool.py    # Searches Coursera/Udemy via SERP API
├── web_extract_tool.py      # Extracts and summarizes web article content
├── linkedin_tool.py         # Formats LinkedIn post from content
├── twitter_tool.py          # Formats Twitter post from content
├── reddit_tool.py           # Formats Reddit post from content

prompt/                      # Prompt templates for agents
└── router_agent_prompt.py   # Prompt logic for router agent

graph.py                    # LangGraph node definitions and wiring
streamlit_app.py            # Streamlit chatbot interface
langgraph_memory.db         # SQLite-based persistent memory storage
README.md                   # Project documentation
```

---

## 🛠️ How It Works

### 🧠 LangGraph Memory System
- `graph.py` defines a LangGraph `StateGraph` with nodes for each agent.
- Uses `SqliteSaver` to persist state and chat memory.
- Memory is stored and retrieved using `thread_id`.
- State includes:
  ```python
  class AgentState(TypedDict):
      query: str
      response: Optional[str]
      intermediate_data: Dict[str, Any]
      next_node: Optional[str]
      messages: Annotated[Sequence[BaseMessage], add_messages]
  ```

### 🔁 Agents
- Each agent is a pure function that takes `query`, `chat_history`, and `intermediate_data`.
- The router decides which agent to invoke based on user input and memory.

### 🤖 Agent Functions
- `topic_learning_agent`: Fetches 5 YouTube + 5 Coursera/Udemy resources.
- `youtube_post_agent`: Generates LinkedIn/Twitter/Reddit posts from YouTube links.
- `article_post_agent`: Does the same but from article links.
- `router_agent`: Uses LLM and prompt logic to determine which agent to invoke.

---

### ✍️ Chat Like a User
```text
User: Find me resources to learn LangGraph
User: Now generate a LinkedIn post from the first video
User: Generate for Reddit too
```
- Each query is routed to the correct agent

---

## 🧠 Memory & Session Management

### 💾 Thread-Based Memory
- Every user session is tied to a `thread_id`
- This ID is used to:
  - Persist chat history
  - Avoid re-summarizing videos/articles
  - Enable time-travel & resuming from checkpoints

### 🧠 View History in Streamlit
In the chatbot interface, all past messages are re-rendered from `langgraph_memory.db` using:
```python
history = checkpointer.get_message_history(thread_id)
```

---

## 🚀 Advanced Capabilities

### ✅ Current Features
- Multi-agent routing with LLM-based decision making
- Persistent memory in SQLite
- Tool-driven agent execution with LangChain
- Streamlit-based chatbot UI

---

## 📚 Technologies Used

| Tool/Library          | Purpose                         |
|-----------------------|---------------------------------|
| LangGraph             | Multi-agent orchestration       |
| LangChain             | LLM interface & tool support    |
| Streamlit             | Chat UI                         |
| SQLite (LangGraph)    | Memory & checkpoints            |
| Gemini & Groq         | LLM providers                   |

---