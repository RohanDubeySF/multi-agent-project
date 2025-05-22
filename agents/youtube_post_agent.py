import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tool')))
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from youtube_extract_tool import generate_youtube_summary
from linkedin_tool import youtube_linkedin_post
from twitter_tool import youtube_twitter_post
from reddit_tool import youtube_reddit_post
from dotenv import load_dotenv
load_dotenv()
from loguru import logger

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

def generate_youtube_content(query:str)->str:
    logger.info(f"Query received:{query}")

    init=initialize_agent(tools=[generate_youtube_summary,youtube_linkedin_post,youtube_twitter_post,youtube_reddit_post],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

    response = init.invoke({"input": {query}})
    logger.debug(response)

    return response['output']


