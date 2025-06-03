from youtube_search import YoutubeSearch
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv("utils/.env")
import json
from loguru import logger


@tool     
def find_youtube_resources(query:str)->json:
    """
    Find YouTube playlists/videos for a given topic
    Returns list of resources with details
    """

    logger.info(f"query: {query}")
    # Search YouTube
    raw_results = YoutubeSearch(f"{query} course OR tutorial OR playlist" ,max_results=5 ).to_json()
    logger.debug(f"raw results fetched:{raw_results}")

    return raw_results

