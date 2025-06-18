import requests
from readability import Document as read_doc
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool,BaseTool
from langchain.chains.summarize import load_summarize_chain
from loguru import logger


class GenerateArticleSummaryTool(BaseTool):
    name :str= "generate_article_summary"
    description :str= "Takes a URL of an article and generates a summary of the content."

    def __init__(self):
        super().__init__()
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.summarizer_model = ChatGroq(model="llama-3.1-8b-instant")

    def _run(self, url: str) -> str:
        try:
            content = self._article_extracter(url)
            summary = self._summarize_content(content)
            logger.debug(f"Summary: {summary}")
            return f"Content Summary:\n{summary}"
        except Exception as e:
            logger.error(f"Error: {e}")

    def _article_extracter(self, url: str) -> str:
        logger.info(f"URL: {url}")

        response = requests.get(url)
        logger.debug(f"Response status: {response.status_code}")

        doc = read_doc(response.text)
        html = doc.summary()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        logger.success("Content extracted")

        return text
    
    def _summarize_content(self, text: str) -> str:
        logger.debug("Running summarization")
        chain = load_summarize_chain(llm=self.summarizer_model, chain_type="map_reduce", verbose=True)
        documents = [Document(page_content=text)]
        summary = chain.invoke(documents)
        return summary["output_text"]


    
