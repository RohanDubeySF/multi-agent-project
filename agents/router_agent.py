from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from agents.topic_learning_agent import learning_agent
from agents.youtube_post_agent import generate_youtube_content
from agents.article_post_agent import article_post_content
from dotenv import load_dotenv
from prompt.router_agent_prompt import router_prompt
load_dotenv()
from loguru import logger

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')


def router_agent(query:str)->str:
    logger.info(f"query: {query}")

    
    prompt=router_prompt

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