from langchain.agents import initialize_agent,AgentType, AgentExecutor
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langchain_core.tools import BaseTool
from typing import List

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.3-70b-versatile')

def my_agent(tools:List[BaseTool])->AgentExecutor:
    return initialize_agent(tools=tools,llm=model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
