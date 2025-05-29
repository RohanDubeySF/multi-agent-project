from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from prompt.twitter_prompt import youtube_twitter_prompt, article_twitter_prompt
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

@tool
def youtube_twitter_post(summary:str)->str:
    """This Tool takes the summary of a youtube video as an Input and Generates a relevant Twitter post"""

    logger.debug(f"summary received {summary}")

    prompt=youtube_twitter_prompt

    logger.info(f"prompt entered \n {prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response

@tool
def article_twitter_post(summary:str)->str:
    """This Tool takes the summary of a article as an Input and Generates a relevant Twitter post"""

    logger.debug(f"summary received {summary}")

    prompt=article_twitter_prompt

    logger.info(f"prompt entered \n{prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response