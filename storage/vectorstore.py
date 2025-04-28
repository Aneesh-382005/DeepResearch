from sentence_transformers import SentenceTransformer
import faiss
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
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.chunkSize = chunkSize
        self.chunkoverlap = chunkoverlap

        self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunkSize , chunk_overlap = chunkoverlap)
        self.metadata: List[Dict[str, Any]] = []
        self.indexPATH = indexPATH

        # Initialize or load index
        if os.path.exists(self.indexPATH):
            try:
                self.index = faiss.read_index(self.indexPATH)
                print(f"Loaded existing index from {self.indexPATH}")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._initialize_index()
        else:
            self._initialize_index()

    def _initialize_index(self, nlist: int = None) -> None:
        """Initialize the FAISS index with the given number of clusters"""
        if nlist is None:
            nlist = self.nlist
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

        embeddings = self.embedder.encode(sampleText).astype("float32")
        n_points = embeddings.shape[0]
        
        # FAISS requires at least 39 * k training points for good clustering
        # We'll set k to be at most n_points / 39
        max_clusters = max(1, n_points // 39)
        if max_clusters < self.nlist:
            print(f"Adjusting clusters from {self.nlist} to {max_clusters} based on available data")
            self._initialize_index(max_clusters)
        
        try:
            self.index.train(embeddings)
            print(f"Successfully trained index with {n_points} points and {max_clusters} clusters")
        except Exception as e:
            print(f"Error during training: {e}")
            # If training fails, fall back to a simpler index
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
            allChunks.extend(chunks)
            allMetadata.extend([metadata.copy() for _ in chunks])

        if not allChunks:
            print("No chunks generated from documents")
            return

        embeddings = self.embedder.encode(allChunks).astype("float32")
        n_points = len(embeddings)
        
        # FAISS requires at least 39 * k training points for good clustering
        max_clusters = max(1, n_points // 39)
        if max_clusters < self.nlist:
            print(f"Adjusting clusters from {self.nlist} to {max_clusters} based on available data")
            self._initialize_index(max_clusters)

        if not self.index.is_trained:
            try:
                self.index.train(embeddings)
                print(f"Successfully trained index with {n_points} points and {max_clusters} clusters")
            except Exception as e:
                print(f"Error during training: {e}")
                # If training fails, fall back to a simpler index
                print("Falling back to IndexFlatL2")
                self.index = faiss.IndexFlatL2(self.dimension)

        try:
            self.index.add(embeddings)
            self.metadata.extend(allMetadata)
            print(f"Successfully added {n_points} vectors to index")
        except Exception as e:
            print(f"Error adding vectors to index: {e}")

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
        