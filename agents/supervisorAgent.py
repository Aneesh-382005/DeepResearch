from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, List, Union
from langchain.schema import AIMessage

import re
import json

from dotenv import load_dotenv
load_dotenv()

import asyncio

import os

moderatorAPIKey = os.getenv("moderatorKey")

template = PromptTemplate(input_variables=["RawQuery"],
                          template = """
You are a prompt moderator for an AI research assistant.  
Your job is to inspect each user query and decide whether it's safe, clear, and sufficiently focused for an automated web-search pipeline.

1. If the query is unclear, overly broad, off-topic, or requests harmful/inappropriate content, do **not** search. Instead, respond with a brief clarifying question or a refusal.

2. Otherwise, output exactly: TOOL_CALL: research_tool with arguments {{"queries": ["your rewritten search-optimized query"]}}

The rewritten query should be concise, well-scoped, and search-optimized. Remove any ambiguous, redundant, or unsafe language.

### Few-Shot Examples:  
User Query: "Tell me everything about AI."  
Response: TOOL_CALL: research_tool with arguments {{"queries": ["Overview of artificial intelligence applications and impact in 2023"]}}

User Query: "How do I hack social media accounts?"  
Response: I'm sorry, but I can't help with activities that could be illegal or harmful.

User Query: "Explain how it works."  
Response: Could you please specify what technology or concept you'd like me to explain?

User Query: "Latest advancements in renewable energy"
Response: TOOL_CALL: research_tool with arguments {{"queries": ["Recent breakthroughs in renewable energy technologies 2023-2025"]}}

User Query: `{RawQuery}`  
Moderator Response:
""")

moderator = ChatGroq(
    model = "llama3-8b-8192", 
    temperature=0,
    api_key = moderatorAPIKey
)

moderatorChain = (
    {"RawQuery": RunnablePassthrough()} | template | moderator
)

async def SuperviseAndResearch(rawQuery: str) -> Union[str, List[Dict[str, Any]]]:

    """
    Moderates the user query and either:
    1. Returns a clarification request if the query is unclear/inappropriate
    2. Calls the research tool with optimized queries if the query is valid
    
    Args:
        raw_query: The user's original query
        
    Returns:
        Either a string with a clarification question or research results
    """
    
    response = await moderatorChain.ainvoke({"RawQuery": rawQuery})

    if isinstance(response, AIMessage):
        moderatorResponse = response.content.strip()
    elif isinstance(response, dict):
        moderatorResponse = list(response.values())[0].strip()
    else:
        moderatorResponse = str(response).strip()

    if moderatorResponse.startswith("TOOL_CALL"):
        try:
            argsMatch = re.search(r'\{.*\}', moderatorResponse)
            if not argsMatch:
                return "Error parsing the search query. Please try again with a clearer question."
            
            argsJSON = argsMatch.group(0)
            args = json.loads(argsJSON)

            queries = args.get("queries", [])
            if not queries:
                return "No valid search queries were generated. Please try again with a more specific question."
            


            from researchTool import researchTool

            print("QUERIES:", queries, type(queries))
            raw = await researchTool.ainvoke({"queries": queries})

            print("RAW:", raw, type(raw))

            if isinstance(raw, AIMessage):
                payload = raw.content
            else:
                payload = raw

            if isinstance(payload, str):
                docs = json.loads(payload)
            else:
                docs = payload
            
            return docs 
        
        except Exception as e:
            return f"An error occurred while processing your query: {e}"
    
    else:
        return moderatorResponse

async def TestQueries():
    testQueries = [
        #"Tell me everything about artificial intelligence and its impact on society.",
        #"How does it work?",
        #"How to hack into a website and steal user data?",
        "Recent advancements in renewable energy technologies in developing countries",
        #"Give me information about controversial political topics in the United States.",
        #"Python programming tutorials for beginners who want to learn data science."
    ]
    
    for query in testQueries:
        result = await SuperviseAndResearch(query)
        print(f"\nOriginal: {query}")
        print(f"Result type: {type(result)}")
        if isinstance(result, str):
            print(f"Clarification: {result}")
        else:
            print(f"Research results: {len(result)} documents retrieved")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(TestQueries())