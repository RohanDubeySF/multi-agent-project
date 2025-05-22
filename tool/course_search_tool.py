from langchain_core.tools import tool
import os
from serpapi import GoogleSearch
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

@tool
def find_online_courses(query:str)-> str:
  """
    Find Online Free Courses from web search for a given topic
    Returns list of resources with details
    """
  logger.info(f"query: {query}")

  params = {
    "engine": "google",
    "q": f"Free {query} site:udemy.com OR site:coursera.org",
    "hl": "en",
    "gl": "in",
    "google_domain": "google.com",
    "num": "5",
    "start": "10",
    "safe": "active",
    "api_key":os.getenv("Serp")
  }
  
  search = GoogleSearch(params)
  results = search.get_json()
  logger.info(f"raw results : \n{results}")

  organic_results = results["organic_results"]
  logger.success("Result ready")
  
  return organic_results

