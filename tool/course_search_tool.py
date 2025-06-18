from langchain_core.tools import BaseTool
import os
from serpapi import GoogleSearch
from loguru import logger
from dotenv import load_dotenv
load_dotenv("utils/.env")

class FindOnlineCoursesTool(BaseTool):
    name :str = "find_online_courses"
    description :str = "Find online free courses for a given topic from Udemy, Coursera, and similar platforms. Returns list of resources with details "

    def _run(self, query: str) -> str:
        try:
            logger.info(f"Query received: {query}")

            params = {
                "engine": "google",
                "q": f"Free {query} site:udemy.com OR site:coursera.org",
                "hl": "en",
                "gl": "in",
                "google_domain": "google.com",
                "num": "5",
                "start": "10",
                "safe": "active",
                "api_key": os.getenv("Serp"),
            }

            search = GoogleSearch(params)
            results = search.get_json()
            logger.debug(f"Raw results: {results}")

            organic_results = results.get("organic_results", [])
            if not organic_results:
                logger.warning("No organic results found")
                return "No courses found."

            logger.success("Results retrieved successfully")
            return organic_results

        except Exception as e:
            logger.error(f"Error during course search: {e}")
            
