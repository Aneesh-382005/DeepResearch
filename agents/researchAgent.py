import sys
import os
import importlib
from typing import List, Dict, Any
import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()

try:
    importlib.import_module('utilities')
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))



from utilities.tavilyModule import CachedTavilyClient
logger = logging.getLogger(__name__)

class TavilyResearchAgent:
    """Agent to perform concurrent Tavily searches and content extraction."""
    def __init__(self, APIkey: str, searchDepth: str = "advanced", maxResults: int = 10, scoreThreshold: float = 0.5, maxConcurrent: int = 5):
        self.client = CachedTavilyClient(APIkey)
        self.searchDepth = searchDepth
        self.maxResults = maxResults
        self.scoreThreshold = scoreThreshold
        self.maxConcurrent = maxConcurrent

    async def runResearch(self, queries: List[str]) -> List[Dict[str, Any]]:
        logger.info("Running %d concurrent searches...", len(queries))

        searchResults = []
        for i in range(0, len(queries), self.maxConcurrent):
            batch = queries[i:i + self.maxConcurrent]
            batchTasks = [
                self.client.search(
                    query = q,
                    searchDepth = self.searchDepth,
                    maxResults = self.maxResults,
                ) for q in batch
            ]

            batchResults = await asyncio.gather(*batchTasks, return_exceptions=True)
            cleaned = [r for r in batchResults if not isinstance(r, Exception)]
            searchResults.extend(cleaned)

        relevantURLs = []
        for result in searchResults:
            for item in result.get("results", []):
                if item.get("score", 0) >= self.scoreThreshold:
                    relevantURLs.append(item.get("url"))
        
        logger.info("Found %d high-quality URLs. Extracting...", len(relevantURLs))

        extractedDocs = []

        for i in range(0, len(relevantURLs), self.maxConcurrent):
            batch = relevantURLs[i:i + self.maxConcurrent]
            batchTasks = [
                self.client.extract(url = u) for u in batch]
            batchResults = await asyncio.gather(*batchTasks, return_exceptions=True)
            extractedDocs.extend([r for r in batchResults if not isinstance(r, Exception)])

        return extractedDocs
