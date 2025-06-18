from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import TypedDict, Optional, Dict, Any, List,Annotated,Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage,HumanMessage
from agents.youtube_post_agent import generate_youtube_content
from agents.topic_learning_agent import learning_agent
from agents.article_post_agent import article_post_content
from agents.router_agent import router_agent
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

class AgentState(TypedDict):
        query:str
        messages: Annotated[Sequence[BaseMessage], add_messages]
        intermediate_data: Optional[Dict[str, Any]]
        next_node:Optional[str]
        response:Optional[str]

class GenAIMultiAgentGraphBuilder:
    def __init__(self):
        self.graph = StateGraph(AgentState)

    def router_node(self, state: AgentState) -> AgentState:
        routing_result = router_agent(
            query=state['query'],
            chat_history=state.get("messages", []),
            intermediate_data=state.get("intermediate_data", {})
        )
        state["query"] = routing_result.get('query', state["query"])
        state["next_node"] = routing_result.get("next_node", "none")
        state["response"] = routing_result["response"]

        for key in routing_result.get("reset_keys", []):
            if key in state["intermediate_data"]:
                state["intermediate_data"][key] = None

        return state

    def youtube_content_node(self, state: AgentState) -> AgentState:
        state['response'], state["intermediate_data"]["last_summary"] = generate_youtube_content(
            query=state['query'],
            intermediate_data=state.get("intermediate_data", {})
        )
        return state

    def course_finder_node(self, state: AgentState) -> AgentState:
        state['response'] = learning_agent(
            query=state['query'],
            intermediate_data=state.get("intermediate_data", {})
        )
        return state

    def article_content_node(self, state: AgentState) -> AgentState:
        state['response'] = article_post_content(
            query=state['query'],
            summary=state.get("intermediate_data", {}).get("last_summary", ""),
            intermediate_data=state.get("intermediate_data", {})
        )
        return state

    def human_review_node(self, state: AgentState) -> AgentState:
        print("🧑‍💻 Awaiting human feedback on content:")
        print("Response:", state.get("response", ""))
        value = interrupt({"text_to_revise": state["response"]})
        return Command(update={"query": f"Content Generated:{state['response']} \n Human Feedback : {value}"})

    def build(self) -> StateGraph:
        self.graph.set_entry_point("Router_Agent")

        self.graph.add_node("Router_Agent", self.router_node)
        self.graph.add_node("youtube_content", self.youtube_content_node)
        self.graph.add_node("article_content", self.article_content_node)
        self.graph.add_node("course_finder", self.course_finder_node)
        self.graph.add_node("human_review", self.human_review_node)

        self.graph.add_conditional_edges(
            source="Router_Agent",
            path=lambda state: state['next_node'],
            path_map={
                "topic_learning_agent": "course_finder",
                "youtube_post_agent": "youtube_content",
                "article_post_agent": "article_content",
                "none": END
            }
        )

        self.graph.add_edge("youtube_content", "human_review")
        self.graph.add_edge("article_content", "human_review")
        self.graph.add_edge("human_review","Router_Agent")

        self.graph.set_finish_point("course_finder")

        return self.graph

class GenAIMultiAgentRunner:
    def __init__(self, db_path: str = "checkpoint.db"):
        sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(sqlite_conn)
        builder = GenAIMultiAgentGraphBuilder()
        self.compiled_graph = builder.build().compile(checkpointer=self.checkpointer)

    def get_app(self):
        return self.compiled_graph