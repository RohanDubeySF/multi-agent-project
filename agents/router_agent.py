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
    print(f" chat history {history_str}")

    prompt=router_prompt

    #logger.info(f"prompt entered to router {prompt.invoke({"query":query,"chat_history":history_str})}")

    chain=prompt | model | JsonOutputParser()
    
    try:
        # Call the chain
        raw_response = chain.invoke({
            "query": query,
            "chat_history": history_str,
            "intermediate_data": str(intermediate_data)
        })
        
        logger.debug(f"[Router Agent] Raw LLM Output: {raw_response}")

        # Validate and fallback
        return {
            "next_node": raw_response.get("next_node", "none"),
            "query": raw_response.get("query", query),
            "reset_keys": raw_response.get("reset_keys", []),
            "response": raw_response.get("response", None)
        }

    except Exception as e:
        logger.error(f"[Router Agent] Failed to parse router output: {e}")
        return {
            "next_node": "none",
            "query": query,
            "reset_keys": [],
            "response": "Sorry, I could not understand that. Can you rephrase?"
        }
