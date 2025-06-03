import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tool')))
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, get_buffer_string
from tool.youtube_extract_tool import generate_youtube_summary
from tool.linkedin_tool import youtube_linkedin_post
from tool.twitter_tool import youtube_twitter_post
from tool.reddit_tool import youtube_reddit_post
from dotenv import load_dotenv
load_dotenv("utils/.env")
from loguru import logger
import ast

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.3-70b-versatile')

def get_tool_output(intermediate_steps, tool_name: str) -> Optional[str]:
    """
    Extracts the output of a specific tool from intermediate steps.
    """
    for action, result in intermediate_steps:
        if action.tool == tool_name:
            return result if isinstance(result, str) else str(result)
    return None
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


def generate_youtube_content(query:str, intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")

    summary=intermediate_data.get("last_summary") or " "

    context = f"""
    You are a YouTube post generation agent. Based on the user query and past conversation, determine what to do:
    - If it's a new link, summarize it and generate social media content.
    - If the user asks for a different platform post (e.g., 'Now do for LinkedIn'), reuse existing summary.

    Use available tools:
    - generate_youtube_summary: Takes a YouTube URL and returns a content summary.
    - youtube_linkedin_post: Takes a summary and generates a LinkedIn post.
    - youtube_twitter_post: Takes a summary and generates a Twitter post.
    - youtube_reddit_post: Takes a summary and generates a Reddit post.

    You must decide the correct flow based on query and history.
    return a structured output with the youtube summary and the post contents 

    Query: {query}
    Existing Summary (May be a Empty string or None based on query):\n{summary}

    output format:
    {{
        "last_summary":"<summary of video>",
        "Post coontent":"The final post for all asked platform in structure"
    }}
    """
    
    init=initialize_agent(tools=[generate_youtube_summary,youtube_linkedin_post,youtube_twitter_post,youtube_reddit_post],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

    response = init.invoke({"input":context})
    output=clean_and_parse_agent_output(response['output'])
    logger.debug(output)

    if intermediate_data is not None:            
        intermediate_data["last_summary"] = output["last_summary"]

    return output["Post coontent"]


