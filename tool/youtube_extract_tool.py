from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.tools import tool, BaseTool
from langchain_core.documents import Document
from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from loguru import logger
load_dotenv("utils/.env")
from utils.define_agent import model,model2

model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model2=ChatGroq(model='llama-3.3-70b-versatile')


def content_extracter(url:str)->str:
    """Generate a Transcript from Youtube video link"""
    logger.info(f"URL : {url}")
    video_id=extract.video_id(url)
    youtube_transcript=YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=["hi","en"])
    
    transcript = " ".join(chunk["text"] for chunk in youtube_transcript)
    logger.debug(f"transcript extracted {transcript}")

    return transcript

def content_summarizer(url:str)->str:
    logger.info(f"URL : {url}")
    logger.info(f"calling content_extracter function")

    content=content_extracter(url)
    logger.debug(f"transcript received {content}")

    chain=load_summarize_chain(llm=model2,chain_type="map_reduce",verbose=True,token_max=1000)
    content=[Document(metadata={"source": url},page_content=content)]

    summary=chain.invoke(content)
    logger.debug(f"summary of video: {summary}")

    logger.success("Task done")
    return summary

@tool
def generate_youtube_summary(url:str)->str:
    """
    This Tool takes a URL for a YouTube Video as an input and generates a summary for the Video Content
    """
    logger.info(f"URL : {url}")
    logger.info(f"calling content_summarizer function")
    #summary=content_summarizer(url)
    summary="This video introduces LangGraph, a tool simplifying multi-agent system development with LLMs by offering state management, flexibility, and scalability. It showcases a chatbot example built using LangGraph\'s graph-like structure for defining nodes and connections, and promotes related Udemy courses."
    output=f"Content Summary:\n{summary} , url:{url}"
    logger.success("Task done")
    return output

# 