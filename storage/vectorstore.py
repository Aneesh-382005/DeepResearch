from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ChunkedVectorStore:
    """
    Vector store that chunks text, embeds with all-MiniLM-L6-v2,
    and uses langchain's FAISS implementation for vector search.
    """
    def __init__(self, 
                modelName: str = "sentence-transformers/all-MiniLM-L6-v2",
                chunkSize: int = 500,
                chunkOverlap: int = 50,
                indexPath: str = "storage/langchain_faiss",
                indexName: str = "index"
                ) -> None:

        print(f"[ChunkedVectorStore __init__] Initializing with FOLDER path: '{indexPath}' and INDEX NAME: '{indexName}'")
        
        self.embedder = HuggingFaceEmbeddings(
            model_name=modelName,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
        )
        print(f"[ChunkedVectorStore __init__] Embedder initialized on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.indexPath = indexPath
        self.indexName = indexName

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize, 
            chunk_overlap=chunkOverlap
        )
        
        self.vectorStore: Optional[FAISS] = None
        
        faissFilePathInFolder = os.path.join(self.indexPath, f"{self.indexName}.faiss")
        pklFilePathInFolder = os.path.join(self.indexPath, f"{self.indexName}.pkl")

        absFaissFilePath = os.path.abspath(faissFilePathInFolder)
        absPklFilePath = os.path.abspath(pklFilePathInFolder)

        print(f"[ChunkedVectorStore __init__] Attempting to load index. Expecting files:")
        print(f"[ChunkedVectorStore __init__]   FAISS file: '{absFaissFilePath}'")
        print(f"[ChunkedVectorStore __init__]   PKL file  : '{absPklFilePath}'")

        if os.path.exists(absFaissFilePath) and os.path.exists(absPklFilePath):
            print(f"[ChunkedVectorStore __init__] Both .faiss and .pkl files found in folder. Attempting FAISS.load_local...")
            try:
                self.vectorStore = FAISS.load_local(
                    allow_dangerous_deserialization = True,
                    folder_path=self.indexPath,
                    embeddings=self.embedder,
                    index_name=self.indexName,
                )
                print(f"[ChunkedVectorStore __init__] Successfully loaded existing index from folder '{self.indexPath}' with index name '{self.indexName}'.")
                if self.vectorStore and self.vectorStore.index:
                    print(f"[ChunkedVectorStore __init__] Loaded index has {self.vectorStore.index.ntotal} vectors.")
                else:
                    print(f"[ChunkedVectorStore __init__] Loaded index, but it appears empty or invalid.")

            except Exception as e:
                print(f"[ChunkedVectorStore __init__] Error during FAISS.load_local from folder '{self.indexPath}', index name '{self.indexName}': {e}")
                logger.error(f"Error loading existing index:", exc_info=True)
                self.vectorStore = None
        else:
            missingFilesDetails = []
            if not os.path.exists(absFaissFilePath):
                missingFilesDetails.append(f"FAISS file missing: {absFaissFilePath}")
            if not os.path.exists(absPklFilePath):
                missingFilesDetails.append(f"PKL file missing: {absPklFilePath}")
            print(f"[ChunkedVectorStore __init__] Could not find required FAISS index files: {'; '.join(missingFilesDetails)}")
            self.vectorStore = None

    def chunkText(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
    def addDocuments(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if not texts:
            print("[ChunkedVectorStore addDocuments] No documents to add")
            return

        if metadatas is None:
            metadatas = [{} for _ in texts]

        allChunks = []
        allMetadatas = []

        for text, metadata in zip(texts, metadatas):
            chunks = self.chunkText(text)
            for i, chunk in enumerate(chunks):
                chunkMetadata = metadata.copy()
                chunkMetadata["sourceDocChunkIndex"] = i
                allChunks.append(chunk)
                allMetadatas.append(chunkMetadata)

        if not allChunks:
            print("[ChunkedVectorStore addDocuments] No chunks generated from documents")
            return

        print(f"[ChunkedVectorStore addDocuments] Adding {len(allChunks)} chunks to the vector store...")
        
        if self.vectorStore is None:
            print(f"[ChunkedVectorStore addDocuments] No existing index. Creating new one in folder '{self.indexPath}' with index name '{self.indexName}'.")
            os.makedirs(self.indexPath, exist_ok=True)
            self.vectorStore = FAISS.from_texts(
                texts=allChunks,
                embedding=self.embedder,
                metadatas=allMetadatas
            )
        else:
            print(f"[ChunkedVectorStore addDocuments] Adding texts to existing index in folder '{self.indexPath}'.")
            self.vectorStore.add_texts(
                texts=allChunks,
                metadatas=allMetadatas
            )
        
        print(f"[ChunkedVectorStore addDocuments] Successfully processed {len(allChunks)} chunks.")
        self.save()

    def search(self, query: str, topK: int = 5) -> List[Tuple[Dict[str, Any], float, str]]:
        if self.vectorStore is None:
            logger.warning("[ChunkedVectorStore search] Search attempted but no FAISS index is loaded or created.")
            return []

        resultsWithScores = self.vectorStore.similarity_search_with_score(query, k=topK)
        
        formattedResults = []
        for docObject, score in resultsWithScores:
            distance = score 
            formattedResults.append((docObject.metadata, distance, docObject.page_content))
            
        return formattedResults
    
    def save(self) -> None:
        if self.vectorStore is None:
            print("[ChunkedVectorStore save] No vector store to save.")
            return
            
        try:
            os.makedirs(self.indexPath, exist_ok=True) 
            
            self.vectorStore.save_local(
                folder_path=self.indexPath,
                index_name=self.indexName,
            )
            print(f"[ChunkedVectorStore save] Successfully saved index to folder '{self.indexPath}' as '{self.indexName}.faiss' and '{self.indexName}.pkl'.")
        except Exception as e:
            print(f"[ChunkedVectorStore save] Error saving index to folder '{self.indexPath}', index name '{self.indexName}': {e}")
            logger.error(f"Error saving index:", exc_info=True)
    
    def toLangchainFaiss(self) -> Optional[FAISS]:
        return self.vectorStore
