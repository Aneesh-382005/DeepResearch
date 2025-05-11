from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
import json
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChunkedVectorStore:
    """
    Vector store that chunks text, embeds with all-MiniLM-L6-v2,
    and uses langchain's FAISS implementation for vector search.
    """
    def __init__(self, 
                modelName: str = "sentence-transformers/all-MiniLM-L6-v2",
                chunkSize: int = 500,
                chunkOverlap: int = 50,
                indexPATH: str = "storage/langchain_faiss") -> None:

        self.embedder = HuggingFaceEmbeddings(
            model_name=modelName,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
        )
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.indexPATH = indexPATH

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize, 
            chunk_overlap=chunkOverlap
        )
        
        # LangChain FAISS vector store
        self.vectorStore = None
        
        # Try to load existing index
        if os.path.exists(f"{indexPATH}.faiss"):
            try:
                self.vectorStore = FAISS.load_local(
                    indexPATH,
                    self.embedder
                )
                print(f"Loaded existing index from {indexPATH}")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self.vectorStore = None

    def chunkText(self, text: str) -> List[str]:
        """Split a long text into overlapping chunks."""
        return self.splitter.split_text(text)
    
    def addDocuments(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Chunk, embed, and add documents to the FAISS index.
        """
        if not texts:
            print("No documents to add")
            return

        if metadatas is None:
            metadatas = [{} for _ in texts]

        allChunks = []
        allMetadatas = []

        for text, metadata in zip(texts, metadatas):
            chunks = self.chunkText(text)
            for i, chunk in enumerate(chunks):
                chunkMetadata = metadata.copy()
                chunkMetadata["chunk"] = chunk
                chunkMetadata["chunkIndex"] = i
                allChunks.append(chunk)
                allMetadatas.append(chunkMetadata)

        if not allChunks:
            print("No chunks generated from documents")
            return

        print(f"Adding {len(allChunks)} chunks to the vector store...")
        
        # If no vector store exists yet, create one
        if self.vectorStore is None:
            self.vectorStore = FAISS.from_texts(
                texts=allChunks,
                embedding=self.embedder,
                metadatas=allMetadatas
            )
        else:
            # Add to existing vector store
            self.vectorStore.add_texts(
                texts=allChunks,
                metadatas=allMetadatas
            )
        
        print(f"Successfully added {len(allChunks)} vectors to index")
        self.save()

    def search(self, query: str, topK: int = 5) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Search the index with a query string and return:
        (metadata, distance, chunk_text) for top_k hits.
        """
        if self.vectorStore is None:
            logger.warning("Search attempted but no FAISS index is loaded or created.")
            return []

        resultsWithScores = self.vectorStore.similarity_search_with_score(query, k=topK)
        
        formattedResults = []
        for doc_object, score in resultsWithScores:
            distance = score 
            formattedResults.append((doc_object.metadata, distance, doc_object.page_content))
            
        return formattedResults
    
    def save(self) -> None:
        """
        Persist the FAISS index to disk.
        """
        if self.vectorStore is None:
            print("No vector store to save")
            return
            
        try:
            os.makedirs(os.path.dirname(self.indexPATH), exist_ok=True)
            self.vectorStore.save_local(self.indexPATH)
            print(f"Saved index to {self.indexPATH}")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def toLangchainFaiss(self):
        """
        Return the langchain FAISS vectorstore for compatibility with LangChain pipelines.
        """
        return self.vectorStore