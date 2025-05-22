from langchain_core.tools import tool
from loguru import logger
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

@tool
def youtube_reddit_post(summary:str)->str:
    """This Tool takes the summary of a youtube video as an Input and Generates a relevant Reddit post"""

    logger.debug(f"summary received {summary}")

    template="""You are helping a user create a Reddit post to share something they recently learned.

    Given a youtube video summary, write a Reddit post that:

    - Starts by explaining that the user recently learned or explored the topic.
    - Summarizes the key concepts or takeaways in clear, digestible language.
    - Explains why the user is sharing this — e.g., to document their learning or contribute to the community.
    - Ends with a question, prompt, or invitation for feedback or resource suggestions.
    - Uses a neutral, informative, and conversational tone that fits Reddit communities like r/learnmachinelearning or r/coding.

    Here is the summary of the YouTube video:
    {summary}

    Write the Reddit post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "Reddit<Reddit logo>:\n<Your Reddit post here>\n\n\n\n"""

    prompt=PromptTemplate(
    template=template,
    input_variables=["summary"])

    logger.info(f"prompt entered \n{prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response

@tool
def article_reddit_post(summary:str)->str:
    """This Tool takes the summary of a article as an Input and Generates a relevant Reddit post"""

    logger.debug(f"summary received {summary}")

    prompt=PromptTemplate(
    template="""You are helping a user create a Reddit post to share something they recently learned.

    Given a article summary, write a Reddit post that:

    - Starts by explaining that the user recently learned or explored the topic.
    - Summarizes the key concepts or takeaways in clear, digestible language.
    - Explains why the user is sharing this — e.g., to document their learning or contribute to the community.
    - Ends with a question, prompt, or invitation for feedback or resource suggestions.
    - Uses a neutral, informative, and conversational tone that fits Reddit communities like r/learnmachinelearning or r/coding.

    Article Summary:
    {summary}

    Write the Reddit post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "Reddit<Reddit logo>:\n\n <Your Reddit post here>\n\n\n\n""",
    input_variables=["summary"])

    logger.info(f"prompt entered \n{prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response