from typing import Optional, Dict, Any
from tool.youtube_extract_tool import generate_youtube_summary
from tool.linkedin_tool import youtube_linkedin_post
from tool.twitter_tool import youtube_twitter_post
from tool.reddit_tool import youtube_reddit_post
from utils.llm_parser import clean_and_parse_agent_output
from utils.define_agent import my_agent
from dotenv import load_dotenv
load_dotenv("utils/.env")
from loguru import logger
from prompt.youtube_agent import youtube_agent_prompt




def generate_youtube_content(query:str, intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")

    summary=intermediate_data.get("last_summary") or " "

    prompt=youtube_agent_prompt.invoke({"query":query,"summary":summary})
    
    init=my_agent([generate_youtube_summary,youtube_linkedin_post,youtube_twitter_post,youtube_reddit_post])

    response = init.invoke({"input":prompt})
    output=clean_and_parse_agent_output(response['output'])
    logger.debug(output)
         
    # intermediate_data["last_summary"] = output["last_summary"]

    return output["Post coontent"],output["last_summary"]
