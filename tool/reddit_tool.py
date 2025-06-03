from langchain_core.tools import tool
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from prompt.reddit_prompt import youtube_reddit_prompt, article_reddit_prompt
from dotenv import load_dotenv
load_dotenv("utils/.env")

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

@tool
def youtube_reddit_post(summary:str)->str:
    """This Tool takes the summary of a youtube video as an Input and Generates a relevant Reddit post"""

    logger.debug(f"summary received {summary}")

    

    prompt=youtube_reddit_prompt

    logger.info(f"prompt entered \n{prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response

@tool
def article_reddit_post(summary:str)->str:
    """This Tool takes the summary of a article as an Input and Generates a relevant Reddit post"""

    logger.debug(f"summary received {summary}")

    prompt=article_reddit_prompt

    logger.info(f"prompt entered \n{prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response= chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response