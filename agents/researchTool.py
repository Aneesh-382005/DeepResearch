from langchain_core.tools import tool
from typing import List, Dict, Any
import asyncio
from researchAgent import TavilyResearchAgent
import os
from dotenv import load_dotenv
load_dotenv()
tavilyAPIKey = os.getenv("tavilyKey")


researchAgent = TavilyResearchAgent(APIkey = tavilyAPIKey)

@tool
async def researchTool(queries: List[str]) -> List[Dict[str, Any]]:
    """
    research_tool(queries: List[str]) → List of extracted docs.
    Crawls web via Tavily, extracts content, and indexes it.
    """
    # Ensure queries are a list of strings
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise ValueError("Input 'queries' must be a list of strings.")
    
    if not queries:
        return []

    results = await researchAgent.runResearch(queries)
    
    return results