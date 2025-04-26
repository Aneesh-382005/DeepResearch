import faiss
from sentence_transformers import SentenceTransformer
import os
import numpy as np

class IVFPQVectorStore:
    def __init__(
             self,
             modelName: str = "nomic-ai/nomic-embed-text-v1",
             dimensions: int = 768,
             nlist: int = 256,
             M: int = 8,
             nbits: int = 8,
             nprobe: int = 10,
             indexPATH: str = "storage/ivfpq.index"
            ):
        self.embedder = SentenceTransformer(modelName, trust_remote_code = True)
        self.dimensions = dimensions
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.index_path = indexPATH

        quantizer = faiss.IndexFlatL2(self.dimensions)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimensions, self.nlist, self.M, self.nbits)
        self.index.nprobe = self.nprobe

        self.metadata = []

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
