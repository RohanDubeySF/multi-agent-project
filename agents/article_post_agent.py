import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tool')))
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from tool.web_extract_tool import generate_article_summary
from tool.linkedin_tool import article_linkedin_post
from tool.twitter_tool import article_twitter_post
from tool.reddit_tool import article_reddit_post
from dotenv import load_dotenv
load_dotenv()
from loguru import logger

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def article_post_content(query:str)->str:
    logger.info(f"Query received:{query}")

    article_agent=initialize_agent(tools=[generate_article_summary,article_linkedin_post,article_twitter_post,article_reddit_post],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION ,verbose=True)
    
    final_query=f"""Create a social media post content for the specified platform mention in the query and if not mentioned Create post for Every plarform tool available \n query : {query}"""
    logger.info(f"Final Query to agent :{final_query}")

    response = article_agent.invoke({"input":final_query})
    logger.debug(response)

    return response['output']

    