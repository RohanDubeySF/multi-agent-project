from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


def supervisor_agent_executer(query:str,agent_output:str)->str:
    template="""
    You are the supervisor agent in a multi-agent system. Your role is to validate, refine, or summarize the output returned by another agent based on the user query.

    1. Ensure the output is complete, relevant, and well-formatted.
    2. If there are missing fields (like Title, URL, or Description), highlight them.
    3. Improve clarity, fix formatting issues, or summarize long responses if needed.

    User Query:
    {query}

    Agent Output:
    {agent_output}

    Your Final Output:
    """
    prompt=PromptTemplate(
        template=template,
        input_variables=["query","agent_output"]
    )

    