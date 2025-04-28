from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any
import torch
import json
import faiss
import numpy as np

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

        self.embedder = HuggingFaceEmbeddings(
            model_name=modelName,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
        )
        self.dimension = dimension
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.chunkSize = chunkSize
        self.chunkoverlap = chunkoverlap

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkoverlap)
        self.metadata: List[Dict[str, Any]] = []
        self.indexPATH = indexPATH
        self.metadataPATH = indexPATH + ".metadata"

        if os.path.exists(self.indexPATH):
            try:
                self.index = faiss.read_index(self.indexPATH)
                print(f"Loaded existing index from {self.indexPATH}")
                if os.path.exists(self.metadataPATH):
                    with open(self.metadataPATH, 'r') as f:
                        self.metadata = json.load(f)
                    print(f"Loaded metadata with {len(self.metadata)} entries")
            except Exception as e:
                print(f"Error loading existing index or metadata: {e}")
                self._initializeIndex()
        else:
            self._initializeIndex()

    def _initializeIndex(self, nlist: int = None) -> None:
        """Initialize the FAISS index with the given number of clusters"""
        if nlist is None:
            nlist = self.nlist
            

        if nlist > 100:
            print(f"Warning: Reducing initial nlist from {nlist} to 100 to avoid training issues")
            nlist = 100
            
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, self.M, self.nbits)
        self.index.nprobe = self.nprobe
        print(f"Initialized new index with {nlist} clusters")

    def chunkText(self, text: str) -> List[str]:
        """Split a long text into overlapping chunks."""
        return self.splitter.split_text(text)
    
    def train(self, sampleText: List[str]) -> None:
        """
        Train IVF centroids + PQ codebooks on sample chunks.
        Must be called before add_documents.
        """
        if not sampleText:
            print("No training data provided")
            return


        embeddings = np.array([self.embedder.embed_query(text) for text in sampleText]).astype("float32")
        nPoints = embeddings.shape[0]

        maxClusters = max(1, min(nPoints // 50, 100))
        
        if maxClusters < self.nlist:
            print(f"Adjusting clusters from {self.nlist} to {maxClusters} based on available data")
            self._initializeIndex(maxClusters)
        
        try:
            print(f"Training index with {nPoints} points and {maxClusters} clusters...")
            self.index.train(embeddings)
            print(f"Successfully trained index")
        except Exception as e:
            print(f"Error during training: {e}")
            print("Falling back to IndexFlatL2")
            self.index = faiss.IndexFlatL2(self.dimension)

    def addDocuments(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Chunk, embed, and add documents to the FAISS index.
        """
        if not texts:
            print("No documents to add")
            return

        allChunks = []
        allMetadata = []

        for text, metadata in zip(texts, metadatas):
            chunks = self.chunkText(text)
            for chunk in chunks:
                chunkMetadata = metadata.copy()
                chunkMetadata["chunk"] = chunk
                allChunks.append(chunk)
                allMetadata.append(chunkMetadata)

        if not allChunks:
            print("No chunks generated from documents")
            return


        batchSize = 32
        allEmbeddings = []
        
        print(f"Embedding {len(allChunks)} chunks in batches of {batchSize}...")
        for i in range(0, len(allChunks), batchSize):
            batch = allChunks[i:i+batchSize]
            batchEmbeddings = [self.embedder.embed_query(chunk) for chunk in batch]
            allEmbeddings.extend(batchEmbeddings)
        
        embeddings = np.array(allEmbeddings).astype("float32")
        nPoints = len(embeddings)
        

        maxClusters = max(1, min(nPoints // 50, 100)) 
        if maxClusters < self.nlist:
            print(f"Adjusting clusters from {self.nlist} to {maxClusters} based on available data")
            self._initializeIndex(maxClusters)

        if not self.index.is_trained:
            try:
                print(f"Training index with {nPoints} points...")
                self.index.train(embeddings)
                print(f"Successfully trained index")
            except Exception as e:
                print(f"Error during training: {e}")
                print("Falling back to IndexFlatL2")
                self.index = faiss.IndexFlatL2(self.dimension)

        try:
            self.index.add(embeddings)
            self.metadata.extend(allMetadata)
            print(f"Successfully added {nPoints} vectors to index")
            self.save()
        except Exception as e:
            print(f"Error adding vectors to index: {e}")

    def search(self, query: str, topK:int = 5) -> List[Dict[str, Any]]:
        """
        Search the index with a query string and return:
        (metadata, distance, chunk_text) for top_k hits.
        """
        queryEmbedding = np.array(self.embedder.embed_query(query)).astype("float32").reshape(1, -1)
        

        if isinstance(self.index, faiss.IndexFlatL2):
            distances, indices = self.index.search(queryEmbedding, topK)
        else:
            self.index.nprobe = min(self.nprobe, self.index.nlist) 
            distances, indices = self.index.search(queryEmbedding, topK)
        
        results = []
        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(self.metadata):
                continue
            metadata = self.metadata[index]
            chunk = metadata.get("chunk", "")
            results.append((metadata, float(distance), chunk))

        return results
    
    def save(self) -> None:
        """
        Persist the FAISS index and metadata to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.indexPATH), exist_ok=True)
            faiss.write_index(self.index, self.indexPATH)
            with open(self.metadataPATH, 'w') as f:
                json.dump(self.metadata, f)
            print(f"Saved index to {self.indexPATH} and metadata to {self.metadataPATH}")
        except Exception as e:
            print(f"Error saving index or metadata: {e}")
        
    def toLangchainFaiss(self):
        """
        Convert to a LangChain FAISS vectorstore for compatibility with LangChain pipelines.
        Note: This creates a new index and copies the vectors.
        """
        allChunks = [meta.get("chunk", "") for meta in self.metadata]
        if not allChunks:
            print("No chunks available to convert")
            return None
        
        batchSize = 32
        allEmbeddings = []
        
        for i in range(0, len(allChunks), batchSize):
            batch = allChunks[i:i+batchSize]
            batchEmbeddings = [self.embedder.embed_query(chunk) for chunk in batch]
            allEmbeddings.extend(batchEmbeddings)

        try:
            langchainFaiss = FAISS.from_embeddings(
                text_embeddings=list(zip(allChunks, allEmbeddings)),
                embedding=self.embedder,
                metadatas=self.metadata
            )
            return langchainFaiss
        except Exception as e:
            print(f"Error converting to LangChain FAISS: {e}")
            return None