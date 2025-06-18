from youtube_search import YoutubeSearch
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
load_dotenv("utils/.env")
import json
from loguru import logger

class YouTubeSearchTool(BaseTool):
    name :str = "youtube_search_tool"
    description :str = "Find YouTube playlists or tutorials for a given topic"

    def __init__(self):
        super().__init__()

    def _run(self, query: str) -> str:
        logger.info(f"Query: {query}")
        try:
            raw_results = YoutubeSearch(f"{query} course OR tutorial OR playlist", max_results=5).to_json()
            logger.debug(f"Raw results fetched: {raw_results}")
            return raw_results
        except Exception as e:
            logger.error(f"Error in YouTubeSearchTool: {e}")
            return json.dumps({"error": str(e)})
        

