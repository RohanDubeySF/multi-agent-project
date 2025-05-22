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
def youtube_linkedin_post(summary:str)->str:
    """This Tool takes the summary of a youtube video as an Input and Generates a relevant LinkedIn post"""

    logger.debug(f"summary received {summary}")

    template="""You are a professional social media content creator.

    Given the summary of a YouTube video, write a LinkedIn post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the video.
    - Maintains a professional and informative tone suitable for a tech-savvy audience
    - Includes 2-3 short bullet points or takeaways.
    - Ends with a relevant call-to-action, like "What are your thoughts?" or "Have you tried this yet?"

    Here is the summary of the YouTube video:
    {summary}

    Write the LinkedIn post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "LinkedIn<LinkedIn logo>:\n<Your LinkedIn post here>\n\n\n\n"""

    prompt=PromptTemplate(
    template=template,
    input_variables=["summary"])

    logger.info(f"prompt entered \n {prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response

@tool
def article_linkedin_post(summary:str)->str:
    """This Tool takes the summary of a article as an Input and Generates a relevant LinkedIn post"""

    logger.debug(f"summary: {summary}")

    template="""You are a professional social media content creator.

    Given the summary of an Article, write a LinkedIn post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the Article
    - Maintains a professional and informative tone suitable for a tech-savvy audience
    - Includes 2-3 short bullet points or takeaways.
    - Ends with a relevant call-to-action, that encourages reader engagement or discussion.
    - strictly follow the above content guidelines and dont mention any other thing which is not present in the context 
    - Dont mention for any youtube video link

    Here is the summary of the Article:
    {summary}

    Write the LinkedIn post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "LinkedIn<LinkedIn logo>:\n<Your LinkedIn post here>\n\n\n\n"""

    prompt=PromptTemplate(
    template=template,
    input_variables=["summary"])

    logger.info(f"prompt entered \n {prompt.invoke(summary)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"summary":summary})
    logger.success("response successfull")

    return response