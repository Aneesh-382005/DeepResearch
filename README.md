# DeepResearch Assistant

DeepResearch Assistant is a prototype system designed to help users explore topics by automatically performing web research and synthesizing information. It takes a user's query, refines it for effective searching, gathers relevant content from the web, and then attempts to generate a concise answer based on the findings.

## Features (Conceptual)

*   **Query Moderation & Optimization:** Initial user queries are assessed for clarity and safety, then rewritten for better search results.
*   **Automated Web Research:** Utilizes web search (via Tavily API) to find relevant online sources.
*   **Content Extraction & Indexing:** Extracts textual content from promising URLs and stores it in a local vector database (FAISS) for quick retrieval.
*   **Information Synthesis:** A language model (powered by Groq Llama3) uses the retrieved information to construct an answer to the original query.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Conda (or Miniconda)
*   Python 3.10 (tested with 3.10.x)
*   Git

### Cloning the Repository

```bash
git clone https://github.com/Aneesh-382005/DeepResearch.git
cd DeepResearch