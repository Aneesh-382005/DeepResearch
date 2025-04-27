from typing import List, Dict, Any
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor

from dotenv import load_dotenv
load_dotenv()

import os
answererAPIKey = os.getenv("answererKey")

import importlib
import sys
try:
    importlib.import_module('storage')
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from storage.vectorstore import ChunkedIVFPQStore
vectorStore = ChunkedIVFPQStore()

def retrieveDocuments(query: str) -> List[Dict[str, Any]]:
    """
    Retrieve documents from the vector store based on the query.
    """
    results = vectorStore.search(query, topK = 5)

    documents = []

    for metadata, distance, chunk in results:
        doc = {
            "PageContent": chunk,
            "Metadata": metadata,
        }
        documents.append(doc)

    return documents

retrievalTool = Tool(
    name = "search", 
    description = "Search for information in the vector database. Input should be a search query.",
    func = retrieveDocuments
)

answerer = ChatGroq(
    model = "llama3-70b-8192", 
    temperature = 0.2,
    api_key = answererAPIKey
)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """You are a helpful assistant that answers questions based on retrieved information.
    Use the search tool to find relevant information before answering.
    Always cite your sources by including references to where you found the information."""),
    ("human", "{input}"),
    ("agent", "{agent_scratchpad}"),
])
    
def createAgent():
    tools = [retrievalTool]

    LLMwithTools = answerer.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
        }
        | prompt
        | LLMwithTools
        | OpenAIFunctionsAgentOutputParser()
    )

    agentExec = AgentExecutor(agent = agent, tools = tools, verbose = True)

    return agentExec

async def answer(question: str) -> Dict[str, Any]:
    """
    Returns structured answer:
      - summary
      - references: list of {url, snippet}
    """
    agent_executor = createAgent()
    result = await agent_executor.ainvoke({"input": question})

    docs = retrieveDocuments(question)

    return {
        "summary": result["output"],
        "references": [
            {
                "url": doc["Metadata"].get("url", "No URL available"),
                "snippet": doc["PageContent"][:200]
            } for doc in docs
        ]
    }