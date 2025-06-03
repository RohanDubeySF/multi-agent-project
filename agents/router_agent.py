from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, get_buffer_string
from agents.topic_learning_agent import learning_agent
from agents.youtube_post_agent import generate_youtube_content
from agents.article_post_agent import article_post_content
from dotenv import load_dotenv
from prompt.router_agent_prompt import router_prompt
load_dotenv("utils/.env")
from loguru import logger
import json

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.3-70b-versatile')


def router_agent(query:str,chat_history: Optional[List[BaseMessage]] = None,
    intermediate_data: Optional[Dict[str, Any]] = None)->json:
    logger.info(f"query: {query}")

    history_str= get_buffer_string(chat_history) if chat_history else " "

    prompt=router_prompt

    #logger.info(f"prompt entered to router {prompt.invoke({"query":query,"chat_history":history_str})}")

    chain=prompt | model | JsonOutputParser()
    try:
        routing_result = chain.invoke({
            "query": query,
            "chat_history": history_str,
            "intermediate_data": str(intermediate_data)
        })
        result = routing_result if isinstance(routing_result, str) else routing_result
    except Exception as e:
        result = {"next_node": "none", "query": query, "reset_keys": []}
    
    logger.info(f"response of Router agent: {result}")

    return result

# def router_agent_executer(query:str)->str:
#     logger.info(f"query: {query}")
    
#     agent=router_agent(query)
#     if agent=="topic_learning_agent":
#         output=learning_agent(query)
#     elif agent=="youtube_post_agent":
#         output=generate_youtube_content(query)
#     elif agent=="article_post_agent":
#         output=article_post_content(query)
#     else:
#         output="Currently this feature is not available, wait for future improvements"

#     return output