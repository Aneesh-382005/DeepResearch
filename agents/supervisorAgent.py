from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

import asyncio

import os

moderatorAPIKey = os.getenv("moderatorKey")

template = PromptTemplate(input_variables=["RawQuery"],
                          template = """
You are a prompt moderator for an AI research assistant.  
Your job is to inspect each user query and decide whether it’s safe, clear, and sufficiently focused for an automated web-search pipeline.  

1. If the query is unclear, overly broad, off-topic, or requests harmful/inappropriate content, do **not** search or answer directly. Instead, respond with a brief clarifying question or a refusal.  
2. Otherwise, rewrite the user’s raw query into a concise, well-scoped, search-optimized prompt. Remove any ambiguous, redundant, or unsafe language.  

### Few-Shot Examples  
**Bad:** “Tell me everything about AI.”  
**Good:** “Overview of AI applications in medical imaging since 2020.”  

**Bad:** “How do I hack social media accounts?”  
**Good:** *(Refuse: “I’m sorry, but I can’t help with that.”)*  

**Bad:** “Explain how it works.”  
**Good:** “How does convolutional neural network feature extraction work in image classification?”  

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

async def supervise(rawQuery: str) -> str:
    response = await moderatorChain.ainvoke({"RawQuery": rawQuery})
    return response.content.strip()

import asyncio

async def testPrompts():
    testQueries = [
        "Tell me everything about artificial intelligence and its impact on society.",
        "How does it work?",
        "How to hack into a website and steal user data?",
        "I want to learn about recent advancements in renewable energy technologies and how they are being implemented in developing countries to combat climate change.",
        "Give me information about controversial political topics in the United States.",
        "Python programming tutorials for beginners who want to learn data science."
    ]
    
    for query in testQueries:
        result = await supervise(query)
        print(f"\nOriginal: {query}")
        print(f"Moderated: {result}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(testPrompts())