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

from storage.vectorstore import ChunkedVectorStore
vectorStore = ChunkedVectorStore()

def retrieveDocuments(query: str) -> List[Dict[str, Any]]:
    """
    Retrieve documents from the vector store based on the query.
    """
    print(f"--- retrieveDocuments called with query: '{query}' ---") 
    results = vectorStore.search(query, topK=5)  # Adjust topK as needed

    print("RESULTS from vectorStore.search:", results, type(results))
    print(f"First result type: {type(results[0]) if results else 'No results'}")
    if results:
        print(f"First result: {results[0]}")

    documents = []
    
    # Handle the actual format returned by your vector store
    for doc in results:
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            # This is a LangChain Document object
            url = doc.metadata.get("url", "No URL available")
            title = doc.metadata.get("title", "No title available") 
            source = doc.metadata.get("source", "No URL available")
            distance = doc.metadata.get("distance", 0.0)  # Default distance
            
            formatted_doc = {
                "pageContent": doc.page_content,
                "metadata": {
                    "url": url,
                    "title": title,
                    "source": source,
                    "distance": distance
                },
            }
            documents.append(formatted_doc)
        else:
            # Fallback for other formats
            print(f"Unexpected document format: {type(doc)}")
            documents.append({
                "pageContent": str(doc),
                "metadata": {
                    "url": "No URL available",
                    "title": "No title available", 
                    "source": "No URL available",
                    "distance": float('inf')
                }
            })

    documents.sort(key=lambda x: x["metadata"].get("distance", float('inf')))
    print(f"--- retrieveDocuments returning {len(documents)} documents ---")
    return documents

# ASYNC version of the retrieval function
async def async_retrieveDocuments(query: str) -> List[Dict[str, Any]]:
    """
    Async version of retrieve documents - wraps the sync function
    """
    import asyncio
    # Run the sync function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retrieveDocuments, query)

# Create tool with BOTH sync and async versions
retrievalTool = Tool(
    name="search", 
    description="Use this tool to search for information to answer user questions. Input should be a concise search query.",
    func=retrieveDocuments,           # Sync version
    coroutine=async_retrieveDocuments # Async version - THIS IS KEY!
)

answerer = ChatGroq(
    model="llama3-70b-8192", 
    temperature=0.1,
    api_key=answererAPIKey
)

def createAgent():
    tools = [retrievalTool]
    
    print(f"Creating agent with tools: {[tool.name for tool in tools]}")
    print(f"Tool descriptions: {[f'{tool.name}: {tool.description}' for tool in tools]}")
    
    # Test tool binding
    LLMwithTools = answerer.bind_tools(tools)
    print(f"LLM bound with tools successfully")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """You are a research assistant. You must ALWAYS use the "search" tool to find information before answering any question.

IMPORTANT: For every user question, you MUST:
1. First use the "search" tool with a relevant search query
2. Then provide a summary based on the search results
3. Never answer without using the search tool first

The search tool will return relevant documents. Use those documents to provide your answer.

If the search returns no results or empty results, say: "I could not find any information on that topic using my search tool."

Do not use any prior knowledge. Only use information from the search tool results."""),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
        }
        | prompt
        | LLMwithTools
        | OpenAIFunctionsAgentOutputParser()
    )

    agentExec = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5,  # Increased iterations
        return_intermediate_steps=True
    )
    return agentExec

async def answer(question: str) -> Dict[str, Any]:
    print(f"Processing question: {question}")
    agentExecutor = createAgent()
    
    try:
        print("Invoking agent...")
        result = await agentExecutor.ainvoke({"input": question})
        print("AGENT RESULT:", result, type(result))
        
        # More detailed logging
        print("Result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
        if isinstance(result, dict):
            print("Output:", result.get("output", "NO OUTPUT KEY"))
            print("Intermediate steps count:", len(result.get("intermediate_steps", [])))
            
    except Exception as e:
        print(f"Error during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return {
            "summary": f"I encountered an error while processing your question: {str(e)}",
            "references": []
        }

    # Better output extraction
    output = result.get("output", "")
    if not output or output.strip() == "":
        print("WARNING: Agent returned empty output!")
        # Look for any content in intermediate steps
        if "intermediate_steps" in result and result["intermediate_steps"]:
            print("Agent executed steps but provided no final output")
            output = "The search was executed but no summary was generated. Please check the references for retrieved information."
        else:
            print("Agent executed no steps")
            output = "The agent did not execute any search steps. Please try rephrasing your question."
    
    summary = output
    references = []

    # Process intermediate steps
    if "intermediate_steps" in result and result["intermediate_steps"]:
        print(f"Processing {len(result['intermediate_steps'])} intermediate steps")
        
        for i, step in enumerate(result["intermediate_steps"]):
            print(f"Step {i}: Type = {type(step)}")
            
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                print(f"  Action: {action}")
                print(f"  Action tool: {getattr(action, 'tool', 'NO TOOL ATTR')}")
                print(f"  Observation type: {type(observation)}")
                print(f"  Observation length: {len(observation) if isinstance(observation, list) else 'Not a list'}")
                
                # Extract references from search tool results
                if hasattr(action, 'tool') and action.tool == "search":
                    if isinstance(observation, list):
                        print(f"--- Extracting references from {len(observation)} documents ---")
                        for doc in observation:
                            if isinstance(doc, dict) and "pageContent" in doc and "metadata" in doc:
                                meta = doc["metadata"]
                                references.append({
                                    "url": meta.get("url", "No URL available"),
                                    "title": meta.get("title", "No title available"),
                                    "source": meta.get("source", "No URL available"),
                                    "distance": meta.get("distance", float('inf')),
                                    "snippet": doc["pageContent"][:200] + "..." if doc["pageContent"] else "No content available"
                                })
                    else:
                        print(f"Unexpected observation format: {type(observation)}")
            else:
                print(f"Unexpected step format: {step}")

    # Sort references by distance
    if references:
        try:
            references.sort(key=lambda x: float(x.get("distance", float('inf'))))
            print(f"Sorted {len(references)} references by distance")
        except (TypeError, ValueError):
            print("Warning: Could not sort references by distance")

    print(f"Final summary: '{summary[:100]}...' (length: {len(summary)})")
    print(f"References found: {len(references)}")

    return {
        "summary": summary,
        "references": references
    }

# Alternative simpler approach that bypasses the agent complexity
async def answer_simple(question: str) -> Dict[str, Any]:
    """
    Simplified approach: directly call tool then ask LLM to summarize
    """
    print(f"Simple approach for question: {question}")
    
    try:
        # Call the async tool directly
        search_results = await async_retrieveDocuments(question)
        print(f"Direct search returned {len(search_results)} results")
        
        if not search_results:
            return {
                "summary": "I could not find any information on that topic using my search tool.",
                "references": []
            }
        
        # Format results for LLM
        formatted_results = "\n\n".join([
            f"Document {i+1}:\nTitle: {doc['metadata']['title']}\nContent: {doc['pageContent'][:500]}..."
            for i, doc in enumerate(search_results[:5])  # Limit to top 5 results
        ])
        
        # Ask LLM to summarize
        summary_prompt = f"""Based on the following search results, provide a comprehensive answer to the question: "{question}"

Search Results:
{formatted_results}

Provide a clear, informative summary based only on the information found in these search results. If the search results don't contain enough information to answer the question, clearly state what information is missing."""

        from langchain_core.messages import HumanMessage
        response = await answerer.ainvoke([HumanMessage(content=summary_prompt)])
        
        # Extract references
        references = []
        for doc in search_results:
            meta = doc["metadata"]
            references.append({
                "url": meta.get("url", "No URL available"),
                "title": meta.get("title", "No title available"),
                "source": meta.get("source", "No URL available"),
                "distance": meta.get("distance", float('inf')),
                "snippet": doc["pageContent"][:200] + "..." if doc["pageContent"] else "No content available"
            })
        
        return {
            "summary": response.content,
            "references": references
        }
        
    except Exception as e:
        print(f"Error in simple approach: {e}")
        import traceback
        traceback.print_exc()
        return {
            "summary": f"Error processing question: {str(e)}",
            "references": []
        }

if __name__ == "__main__":
    import asyncio
    print(f"Instance's configured index path: {vectorStore.indexPath}")
    if vectorStore.vectorStore is None:
        print("WARNING: Vector store was not loaded successfully during initialization.")
    else:
        print(f"Vector store seems to be loaded. Index has {vectorStore.vectorStore.index.ntotal} vectors.")
    
    # FIRST: Test the tool directly
    print("\n=== TESTING TOOL DIRECTLY ===")
    try:
        direct_results = retrieveDocuments("renewable energy")
        print(f"Direct tool test returned {len(direct_results)} documents")
        if direct_results:
            print(f"First document preview: {direct_results[0]['pageContent'][:100]}...")
        else:
            print("No documents returned from direct tool test!")
    except Exception as e:
        print(f"Direct tool test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # SECOND: Test LLM tool calling capability
    print("\n=== TESTING LLM TOOL CALLING ===")
    try:
        llm_with_tools = answerer.bind_tools([retrievalTool])
        from langchain_core.messages import HumanMessage
        
        test_message = HumanMessage(content="Use the search tool to find information about renewable energy")
        response = asyncio.run(llm_with_tools.ainvoke([test_message]))
        print(f"LLM response type: {type(response)}")
        print(f"LLM response: {response}")
    except Exception as e:
        print(f"LLM tool calling test failed: {e}")
        import traceback
        traceback.print_exc()
    
    test_question = "Summarize findings on renewable energy."
    
    print(f"\n=== TESTING AGENT APPROACH ===")
    response1 = asyncio.run(answer(test_question))
    print("\n--- AGENT RESPONSE ---")
    print("Summary:", response1["summary"])
    print(f"References: {len(response1['references'])}")
    
    print(f"\n=== TESTING SIMPLE APPROACH ===")
    response2 = asyncio.run(answer_simple(test_question))
    print("\n--- SIMPLE RESPONSE ---")  
    print("Summary:", response2["summary"])
    print(f"References: {len(response2['references'])}")