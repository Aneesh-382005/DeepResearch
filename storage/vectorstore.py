from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any
import torch



class ChunkedIVFPQStore:
    """
    Vector store that chunks text, embeds with all-MiniLM-L6-v2,
    and indexes with FAISS IndexIVFPQ for ANN search.
    """
    def __init__(self, 
                modelName: str = "sentence-transformers/all-MiniLM-L6-v2",
                dimension: int = 384,
                nlist: int = 256,
                M: int = 8,
                nbits: int = 8,
                nprobe: int = 10,
                chunkSize: int = 500,
                chunkoverlap: int = 50,
                indexPATH: str = "storage/ivfpq.faiss") -> None:
        self.embedder = SentenceTransformer(modelName, device = "cuda" if torch.cuda.is_available() else "cpu")
        self.dimension = dimension
        
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, M, nbits)
        self.index.nprobe = nprobe

        self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunkSize , chunk_overlap = chunkoverlap)

        self.metadata: List[Dict[str, Any]] = []

        self.indexPATH = indexPATH
        if os.path.exists(self.indexPATH):
            self.index = faiss.read_index(self.indexPATH)

    def chunkText(self, text: List[str]) -> List[str]:
        """Split a long text into overlapping chunks."""
        return self.splitter.split_text(text)
    
    def train(self, sampleText: List[str]) -> None:
        """
        Train IVF centroids + PQ codebooks on sample chunks.
        Must be called before add_documents.
        """
        embeddings = self.embedder.encode(sampleText).astype("float32")
        self.index.train(embeddings)

    def addDocuments(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Chunk, embed, and add documents to the FAISS index.
        """
        allChunks = []
        allMetadata = []

        for text, metadata in zip(texts, metadatas):
            chunks = self.chunkText(text)
            allChunks.extend(chunks)
            allMetadata.extend([metadata.copy() for _ in chunks])

        embeddings = self.embedder.encode(allChunks).astype("float32")
        if not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(allMetadata)

    def search(self, query: str, topK:int = 5) -> List[Dict[str, Any]]:
        """
        Search the index with a query string and return:
        (metadata, distance, chunk_text) for top_k hits.
        """
        queryEmbedder = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(queryEmbedder, topK)
        
        results = []
        for distance, index in zip(distances[0], indices[0]):
            metadata = self.metadata[index]
            chunk = metadata.get("chunk", "")
            results.append((metadata, float(distance), chunk))

        return results
    
    def save(self) -> None:
        """
        Persist the FAISS index to disk.
        """
        faiss.write_index(self.index, self.indexPATH)
        