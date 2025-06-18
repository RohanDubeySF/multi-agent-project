from typing import Optional, Dict, Any, List
from tool.web_extract_tool import GenerateArticleSummaryTool
from tool.linkedin_tool import article_linkedin_post
from tool.twitter_tool import article_twitter_post
from tool.reddit_tool import article_reddit_post
from utils.llm_parser import clean_and_parse_agent_output
from utils.define_agent import my_agent
from dotenv import load_dotenv
load_dotenv("utils/.env")
from loguru import logger
from prompt.article_agent import article_agent_prompt

def article_post_content(query:str,intermediate_data: Optional[Dict[str, Any]] = None)->str:
    logger.info(f"Query received:{query}")

    summary=intermediate_data.get("last_summary") or " "

    prompt=article_agent_prompt.invoke({"query":query,"summary":summary})

    article_agent=my_agent(tools=[GenerateArticleSummaryTool(),article_linkedin_post,article_twitter_post,article_reddit_post])
    
    logger.info(f"Final Query to agent :{prompt}")

    response = article_agent.invoke({"input":prompt})

    output=clean_and_parse_agent_output(response['output'])
    logger.debug(output)

    if intermediate_data is not None:            
        intermediate_data["last_summary"] = output["last_summary"]
    

    return output["Post content"]


    