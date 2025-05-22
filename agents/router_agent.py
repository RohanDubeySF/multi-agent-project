from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from topic_learning_agent import learning_agent
from youtube_post_agent import generate_youtube_content
from article_post_agent import article_post_content
from dotenv import load_dotenv
load_dotenv()
from loguru import logger

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')


def router_agent(query:str)->str:
    logger.info(f"query: {query}")

    template = """You are a multi-agent router responsible for selecting the most suitable agent to handle the user's query. Choose one agent from the list below based on the task described.

    Available Agents:
    - topic_learning_agent: Fetches learning resources such as YouTube videos, playlists, and online courses based on a topic provided by the user.
    - youtube_post_agent: Generates social media posts (LinkedIn, Twitter, Reddit) based on the content of a provided YouTube video link.
    - article_post_agent: Generates social media posts (LinkedIn, Twitter, Reddit) based on the content of a provided web article link.

    Routing Logic:
    1. If the user provides a **topic they want to learn about**, and is asking for **learning resources**, route to **topic_learning_agent**.
    2. If the user provides a **YouTube link** and wants to generate social media content from it, route to **youtube_post_agent**.
    3. If the user provides a **web article link** or refers to an online article and wants to generate social media content, route to **article_post_agent**.

    User Query:
    {query}

    Output Format:
    Return ONLY the most suitable agent name from the list above. Do not explain your choice. If the intent is unclear or If none match clearly, return: `none`.
    """
    prompt=PromptTemplate(
        template=template,
        input_variables=["query"]
    )

    logger.info(f"prompt entered to router {prompt.invoke(query)}")

    chain=prompt | model | StrOutputParser()
    response=chain.invoke({"query":query})
    logger.info(f"response of Router agent: {response}")

    return response

def router_agent_executer(query:str)->str:
    logger.info(f"query: {query}")
    
    agent=router_agent(query)
    if agent=="topic_learning_agent":
        output=learning_agent(query)
    elif agent=="youtube_post_agent":
        output=generate_youtube_content(query)
    elif agent=="article_post_agent":
        output=article_post_content(query)
    else:
        output="Currently this feature is not available, wait for future improvements"

    return output