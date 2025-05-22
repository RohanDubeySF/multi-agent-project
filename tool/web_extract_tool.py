import requests
from readability import Document as read_doc
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from loguru import logger

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.1-8b-instant')

def article_extracter(url:str)->str:
    logger.info(f"URL : {url}")

    response = requests.get(url)
    logger.debug(f"response status: {response.status_code}")
    doc = read_doc(response.text)

    html = doc.summary()
    
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    logger.debug(f"Content :{text}")
    logger.success("Content Extracted ")
    return text



def content_summarizer(text:str)->str:
    logger.debug(f"content Received for summary: {text}")
    chain=load_summarize_chain(llm=model2,chain_type="map_reduce",verbose=True)
    content=[Document(page_content=text)]
    summary=chain.invoke(content)
    return summary

@tool
def generate_article_summary(url:str)->str:
    """
    This Tool takes a URL of an Article as an input and generates a summary of the Page Content 
    """
    logger.info(f"URL : {url}")

    content=article_extracter(url)
    summary=content_summarizer(text=content)
    logger.debug(f"Summary received:{summary}")
    output=f"Content Summary:\n{summary}"

    return output

    
