import re
from loguru import logger
from dotenv import load_dotenv
load_dotenv("utils/.env")
from typing import Optional,List,Dict,Any
from tool.youtube_search_tool import YouTubeSearchTool 
from tool.course_search_tool import FindOnlineCoursesTool
from utils.define_agent import my_agent
from prompt.learning_agent import learning_agent_prompt


def learning_agent(query:str,intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")


    prompt=learning_agent_prompt.invoke({"query":query})

    agent=my_agent(tools=[YouTubeSearchTool(),FindOnlineCoursesTool()])
    
    logger.info("[Learning Agent] Executing LLM with toolchain...")

    response = agent.invoke({"input": prompt})

    urls = re.findall(r'https?://[^\s]+', response['output'])
    youtube_links = [url for url in urls if "youtube.com" in url or "youtu.be" in url]

    if intermediate_data is not None:
        intermediate_data["youtube_links"] = youtube_links
        
    logger.debug(response)

    return response['output']
  