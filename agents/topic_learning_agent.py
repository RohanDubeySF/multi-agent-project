import sys
import os
from langchain.agents import initialize_agent,AgentType
from langchain.prompts import PromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tool')))
 
from youtube_search_tool import find_youtube_resources  
from course_search_tool import find_online_courses

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

def learning_agent(query:str,model:str=model)->str:
    logger.info(f"Query received:{query}")

    format="""Extract the following details from YouTube search results and web search:\n"
                "1. Title\n"
                "2. Channel Name\n"
                "3. URL\n"
                "4. Duration (if available)\n"
                "5. Brief description (if available) \n"
                "Format as a numbered list with clear labels for each field.\n"
                "Include only the most relevant 5 results from both Youtube as well as web search respectively"""

    agent=initialize_agent(tools=[find_youtube_resources,find_online_courses],llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    final_query=f"Find me learning resources for {query} and \n {format}"
    logger.info(f"Final Query to agent :{final_query}")

    response = agent.invoke({"input": final_query})

    logger.debug(response)

    return response['output']
  