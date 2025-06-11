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
import ast 

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def clean_and_parse_agent_output(agent_output: str) -> dict:
    # Step 1: Strip markdown-style code block markers
    cleaned = agent_output.strip()
    
    # Remove starting ```json or ``` if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    
    # Remove ending ``` if present
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    
    # Step 2: Parse the string as a Python dict
    return ast.literal_eval(cleaned)

def article_post_content(query:str,intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")

    summary=intermediate_data.get("last_summary") or " "

    context = f"""
    You are an article-based social media post generation agent.

    Your goals:
    - Decide whether to summarize the article using generate_article_summary
    - Then generate post(s) using appropriate tool: article_linkedin_post, article_twitter_post, or article_reddit_post
    - If summary already exists in memory, reuse it instead of generating again
    - Choose the platform(s) based on user query or default to all if not specified

    Query: {query}
    Existing Summary (May be a Empty string or None based on query):\n{summary}

    output format:
    {{
        "last_summary":"<summary of video>",
        "Post content":"The final post for all asked platform in structure"
    }}
    """

    article_agent=initialize_agent(tools=[generate_article_summary,article_linkedin_post,article_twitter_post,article_reddit_post],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION ,verbose=True)
    
    logger.info(f"Final Query to agent :{context}")

    response = article_agent.invoke({"input":context})

    output=clean_and_parse_agent_output(response['output'])
    logger.debug(output)

    if intermediate_data is not None:            
        intermediate_data["last_summary"] = output["last_summary"]
    

    return output["Post content"]


    