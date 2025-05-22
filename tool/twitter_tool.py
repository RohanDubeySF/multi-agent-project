from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

@tool
def youtube_twitter_post(summary:str)->str:
    """This Tool takes the summary of a youtube video as an Input and Generates a relevant Twitter post"""

    logger.debug(f"summary received {summary}")

    prompt=PromptTemplate(
    template="""You are a professional social media content creator.

    Given the summary of a YouTube video, write a Twitter post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the video in simple terms.
    - Keeps the post under 200 characters.
    - Ends with a relevant call-to-action, that encourages reader engagement or discussion.
    - Also include the link of the video

    Here is the summary of the YouTube video:
    {summary}

   Write the tweet in first person, as if the user is sharing their own learning.
    output Format:\n
    "Twitter<twitter logo>:\n\n <Your Twitter post here>\n\n\n\n""",
    input_variables=["summary"])

    logger.info(f"prompt entered \n {prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response

@tool
def article_twitter_post(summary:str)->str:
    """This Tool takes the summary of a article as an Input and Generates a relevant Twitter post"""

    logger.debug(f"summary received {summary}")

    prompt=PromptTemplate(
    template="""You are a professional social media content creator.

    Given the summary of an Article, write a Twitter post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the Article in simple terms.
    - Keeps the post under 200 characters.
    - Ends with a relevant call-to-action, that encourages reader engagement or discussion.

    Here is the summary of the Article:
    {summary}

    Write the tweet in first person, as if the user is sharing their own learning.
    output Format:\n
    "Twitter<twitter logo>:\n\n <Your Twitter post here>\n\n\n\n""",
    input_variables=["summary"])

    logger.info(f"prompt entered \n{prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response