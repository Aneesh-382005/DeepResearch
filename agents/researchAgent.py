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
from storage.vectorstore import ChunkedVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TavilyResearchAgent:
    def __init__(self, APIkey: str, searchDepth: str = "advanced", maxResults: int = 10, scoreThreshold: float = 0.5, maxConcurrent: int = 5):
        self.client = CachedTavilyClient(APIkey)
        self.searchDepth = searchDepth
        self.maxResults = maxResults
        self.scoreThreshold = scoreThreshold
        self.maxConcurrent = maxConcurrent
        self.vectorStore = ChunkedVectorStore()

    async def runResearch(self, queries: List[str]) -> List[Dict[str, Any]]:
        logger.info("Running %d Tavily searches for queries: %s", len(queries), queries)

        searchResults = []
        searchExceptions = []
        for i in range(0, len(queries), self.maxConcurrent):
            batch = queries[i:i + self.maxConcurrent]
            batchTasks = [
                self.client.search(
                    query=q,
                    searchDepth=self.searchDepth,
                    maxResults=self.maxResults,
                ) for q in batch
            ]
            batchResultsWithExceptions = await asyncio.gather(*batchTasks, return_exceptions=True)
            for res in batchResultsWithExceptions:
                if isinstance(res, Exception):
                    searchExceptions.append(res)
                    logger.warning(f"Tavily search failed for a query in batch {batch}: {res}")
                else:
                    searchResults.append(res)
        
        if searchExceptions:
            logger.warning(f"Encountered {len(searchExceptions)} exceptions during Tavily search.")

        relevantUrls = []
        for result in searchResults:
            if result and "results" in result:
                for item in result["results"]:
                    if item.get("score", 0) >= self.scoreThreshold and item.get("url"):
                        relevantUrls.append(item.get("url"))
        uniqueRelevantUrls = sorted(list(set(relevantUrls)))
        logger.info("Found %d unique, high-quality URLs. Extracting content...", len(uniqueRelevantUrls))

        if not uniqueRelevantUrls:
            logger.info("No relevant URLs found to extract content from.")
            return []
        
        extractedDocsFromTavily = []
        extractionExceptions = []
        for i in range(0, len(uniqueRelevantUrls), self.maxConcurrent):
            batchUrls = uniqueRelevantUrls[i:i + self.maxConcurrent]
            batchTasks = [self.client.extract(url=u) for u in batchUrls]
            
            batchResultsWithExceptions = await asyncio.gather(*batchTasks, return_exceptions=True)
            for res in batchResultsWithExceptions:
                if isinstance(res, Exception):
                    extractionExceptions.append(res)
                    logger.warning(f"Tavily content extraction failed for a URL: {res}")
                elif res:
                    extractedDocsFromTavily.append(res)
        
        if extractionExceptions:
            logger.warning(f"Encountered {len(extractionExceptions)} exceptions during content extraction.")

        textsForVectorStore = []
        metadatasForVectorStore = []

        for docContent in extractedDocsFromTavily:
            if not docContent or not isinstance(docContent, dict):
                logger.warning(f"Skipping invalid document content: {docContent}")
                continue
                        
            url = docContent.get("url", "No URL available")
            for contentItem in docContent.get("results", []): 
                if not contentItem or not isinstance(contentItem, dict):
                    continue
                            
                rawContent = contentItem.get("raw_content")
                if rawContent:
                    textsForVectorStore.append(rawContent)
                    metadatasForVectorStore.append({
                        "url": url,
                        "title": contentItem.get("title", "No title available"),
                        "source": contentItem.get("source", url)
                    })

        if textsForVectorStore and metadatasForVectorStore:
            logger.info(f"Adding {len(textsForVectorStore)} processed content pieces to vector store.")
            self.vectorStore.addDocuments(textsForVectorStore, metadatasForVectorStore)
        else:
            logger.info("No text content extracted to add to the vector store.")
        
        return extractedDocsFromTavily
