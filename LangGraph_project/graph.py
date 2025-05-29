from langgraph.graph import StateGraph, END
from typing import TypedDict,Optional

from agents.youtube_post_agent import generate_youtube_content
from agents.topic_learning_agent import learning_agent
from agents.article_post_agent import article_post_content
from agents.router_agent import router_agent


class AgentState(TypedDict):
    query:str
    next_node:Optional[str]
    response:Optional[str]


graph=StateGraph(AgentState)

def youtube_content_node(state:AgentState)-> AgentState:
    state['response']= generate_youtube_content(state['query'])
    return state
  

def course_finder_node(state:AgentState)-> AgentState:
    state['response']=learning_agent(state['query'])
    return state


def article_content_node(state:AgentState)->AgentState:
    state['response']=article_post_content(state['query'])
    return state


graph.set_entry_point("Router_Agent")


def router_node(state:AgentState)-> AgentState:
    state['next_node']=router_agent(state['query'])
    
    return state


graph.add_node("Router_Agent",router_node)
graph.add_node("youtube_content",youtube_content_node)
graph.add_node("course_finder",course_finder_node)
graph.add_node("article_content",article_content_node)


graph.add_conditional_edges(
    source="Router_Agent",
    path=lambda state : state['next_node'],
    path_map={
        "topic_learning_agent":"course_finder",
        "youtube_post_agent":"youtube_content",
        "article_post_agent":"article_content"
    }
)


graph.set_finish_point("youtube_content")
graph.set_finish_point("course_finder")
graph.set_finish_point("article_content")


app=graph.compile()


from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))


inputs={"query":"create content for https://www.youtube.com/watch?v=BGTx91t8q50"}


result=app.invoke(inputs)


print(result)


print(result["response"])




