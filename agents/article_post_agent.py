import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tool')))
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, get_buffer_string
from tool.web_extract_tool import generate_article_summary
from tool.linkedin_tool import article_linkedin_post
from tool.twitter_tool import article_twitter_post
from tool.reddit_tool import article_reddit_post
from dotenv import load_dotenv
load_dotenv("utils/.env")
from loguru import logger

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def get_tool_output(intermediate_steps, tool_name: str) -> Optional[str]:
    """Utility to extract output of a specific tool from agent trace."""
    for action, result in intermediate_steps:
        if action.tool == tool_name:
            return result if isinstance(result, str) else str(result)
    return None

def article_post_content(query:str,intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")

    intermediate_data.get("last_summary") or ""

    context = f"""
    You are an article-based social media post generation agent.

    Your goals:
    - Decide whether to summarize the article using generate_article_summary
    - Then generate post(s) using appropriate tool: article_linkedin_post, article_twitter_post, or article_reddit_post
    - If summary already exists in memory, reuse it instead of generating again
    - Choose the platform(s) based on user query or default to all if not specified

    Query: {query}
    Existing Summary (May be a Empty string or None based on query):\n{summary}
    """

    article_agent=initialize_agent(tools=[generate_article_summary,article_linkedin_post,article_twitter_post,article_reddit_post],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION ,verbose=True,return_intermediate_steps=True)
    
    logger.info(f"Final Query to agent :{context}")

    response = article_agent.invoke({"input":context})
    intermediate_steps = response.get("intermediate_steps", [])
    logger.debug(response)

    if intermediate_data is not None:
        article_summary = get_tool_output(intermediate_steps, "generate_article_summary")
        if article_summary:
            intermediate_data["last_summary"] = article_summary
            logger.info("[Article Post Agent] Summary saved to memory.")

        intermediate_data["last_post"] = response['output']
        logger.success("[Article Post Agent] Post content saved to memory.")

    return response['output']

    