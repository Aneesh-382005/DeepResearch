# DeepResearch

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
```
### Setup

Create and activate a Conda environment:
```bash
conda create -n deepresearch_env python=3.10 -y
conda activate deepresearch_env
```

### Install dependencies:
Make sure you have a requirements.txt file in your project root. If not, create one with pip freeze > requirements.txt after installing all necessary packages in your `deepresearch_env`.
```bash
pip install -r requirements.txt
```
### Set up API Keys:
This project requires API keys for external services.
Create a .env file in the root of the project directory. You can copy the structure from example.env if you create one, or make it from scratch.

Add your API keys to the .env file:
```
moderatorKey = YOUR_GROQ_API_KEY_FOR_MODERATOR
answererKey = YOUR_GROQ_API_KEY_FOR_ANSWERER
tavilyKey = YOUR_TAVILY_API_KEY
```

Important: Ensure your `.env` file is listed in your `.gitignore` file to prevent accidentally committing your secret keys.



### Example Flow for now

The system is composed of different agents. A typical flow might involve:
Running the `supervisorAgent.py` to perform research on a query and populate the vector store:

```
python agents/supervisorAgent.py
```
(You may want to modify the test queries within this script before running.)
Running the answerAgent.py to ask questions against the researched information:
```
python agents/answerAgent.py
```
(Modify the test question within this script to query your indexed data.)


### Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. **Fork the Project**
2. **Create your Feature Branch (`git checkout -b feature/AmazingFeature`)**
3. **Commit your Changes (`git commit -m 'Add some AmazingFeature'`)**
4. **ush to the Branch (`git push origin feature/AmazingFeature`)**
5. **Open a Pull Request**

### License
Distributed under the MIT License. See `LICENSE.txt` for more information.