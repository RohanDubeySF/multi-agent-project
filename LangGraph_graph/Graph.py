from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import TypedDict, Optional, Dict, Any, List,Annotated,Sequence
from langchain_core.messages import BaseMessage
from agents.youtube_post_agent import generate_youtube_content
from agents.topic_learning_agent import learning_agent
from agents.article_post_agent import article_post_content
from agents.router_agent import router_agent
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3 
from langgraph.store.sqlite import SqliteStore

def graph():
        
    class AgentState(TypedDict):
        query:str
        messages: Annotated[Sequence[BaseMessage], add_messages]
        intermediate_data: Dict[str, Any]
        next_node:Optional[str]
        response:Optional[str]

    def router_node(state:AgentState)-> AgentState:
        routing_result=router_agent(query=state['query'],chat_history=state.get("chat_history", []),intermediate_data=state.get("intermediate_data", {}))
        state["query"]=routing_result.get('query',state["query"])
        state["next_node"] = routing_result.get("next_node", "none")

        for key in routing_result.get("reset_keys", []):
            if key in state["intermediate_data"]:
                state["intermediate_data"][key] = None
        return state

    def youtube_content_node(state:AgentState)-> AgentState:
        state['response']= generate_youtube_content(query=state['query'],intermediate_data=state.get("intermediate_data", {}))
        return state
            

    def course_finder_node(state:AgentState)-> AgentState:
        state['response']=learning_agent(query=state['query'],intermediate_data=state.get("intermediate_data", {}))
        return state


    def article_content_node(state:AgentState)->AgentState:
        state['response']=article_post_content(query=state['query'],summary=state.get(['intermediate_data']["last_summary"],""),intermediate_data=state.get("intermediate_data", {}))
        return state
    
    
    graph=StateGraph(AgentState)


    graph.set_entry_point("Router_Agent")

    graph.add_node("Router_Agent",router_node)
    graph.add_node("youtube_content",youtube_content_node)
    graph.add_node("article_content",article_content_node)
    graph.add_node("course_finder",course_finder_node)


    graph.add_conditional_edges(
        source="Router_Agent",
        path=lambda state : state['next_node'],
        path_map={
            "topic_learning_agent":"course_finder",
            "youtube_post_agent":"youtube_content",
            "article_post_agent":"article_content",
            "None":END
        }
    )


    graph.set_finish_point("youtube_content")
    graph.set_finish_point("course_finder")
    graph.set_finish_point("article_content")


    sqlite_conn=sqlite3.connect("checkpoint.sqlite",check_same_thread=False)
    checkpointer=SqliteSaver(sqlite_conn)

    store=SqliteStore.from_conn_string("sqlite:///langgraph_store.db")

    return graph.compile(checkpointer=checkpointer,store=store)




