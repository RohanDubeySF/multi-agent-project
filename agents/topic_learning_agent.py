import re
from langchain.agents import initialize_agent,AgentType
from langchain.prompts import PromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from loguru import logger
from dotenv import load_dotenv
load_dotenv("utils/.env")
from typing import Optional,List,Dict,Any
from langchain_core.messages import BaseMessage, get_buffer_string
from tool.youtube_search_tool import find_youtube_resources  
from tool.course_search_tool import find_online_courses

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

def learning_agent(query:str,intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")


    context = f"""
    You are a learning resource recommender agent. Based on the query, determine the best tools to use.
    Available tools:
    - YouTube search for tutorials
    - Online course search (Udemy/Coursera)

    Format your response with this structure:
    1. Title
    2. Channel/Source
    3. URL
    4. Duration (Only if available)
    5. Brief description (Only if available)

    Include up to 5 top results per source (YouTube & Web).
    Query: {query}
    """

    agent=initialize_agent(tools=[find_youtube_resources,find_online_courses],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    logger.info("[Learning Agent] Executing LLM with toolchain...")

    response = agent.invoke({"input": context})

    urls = re.findall(r'https?://[^\s]+', response['output'])
    youtube_links = [url for url in urls if "youtube.com" in url or "youtu.be" in url]

    if intermediate_data is not None:
        intermediate_data["youtube_links"] = youtube_links
        
    logger.debug(response)

    return response['output']
  