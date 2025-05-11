import logging
from pathlib import Path
from typing import List, Optional

# Embeddings and Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector embeddings and retrieval"""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            logger.info(f"Successfully initialized HuggingFaceEmbeddings with model: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEmbeddings model '{embedding_model_name}': {e}", exc_info=True)
            raise # Re-raise the exception to halt initialization if embeddings can't load

        self.vector_store: Optional[FAISS] = None # Type hint for clarity

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a vector store from documents"""
        if not documents:
            logger.error("No documents provided to create vector store.")
            raise ValueError("No documents provided to create vector store")

        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created FAISS vector store with {len(documents)} processed document chunks.")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store: {e}", exc_info=True)
            raise

    def save_vector_store(self, path: str = "./vector_store"):
        """Save the vector store to disk"""
        if self.vector_store:
            try:
                vector_store_path = Path(path)
                vector_store_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                self.vector_store.save_local(str(vector_store_path))
                logger.info(f"Saved vector store to {vector_store_path}")
            except Exception as e:
                logger.error(f"Failed to save vector store to {path}: {e}", exc_info=True)
        else:
            logger.warning("Attempted to save vector store, but it has not been created yet.")

    def load_vector_store(self, path: str = "./vector_store") -> Optional[FAISS]:
        """Load the vector store from disk"""
        vector_store_path = Path(path)
        if vector_store_path.exists() and vector_store_path.is_dir(): # Check if path is a directory
            try:
                # IMPORTANT: Allow dangerous deserialization only if you trust the source
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True # Be aware of the security implications
                )
                logger.info(f"Loaded vector store from {vector_store_path}")
                return self.vector_store
            except Exception as e:
                logger.error(f"Failed to load vector store from {vector_store_path}: {e}. It might be corrupted or incompatible.", exc_info=True)
                self.vector_store = None # Ensure it's None on failure
                return None # Return None if loading fails
        else:
            logger.info(f"Vector store not found at {vector_store_path}. A new one may need to be created.")
            self.vector_store = None
            return None

    def get_retriever(self, k: int = 3) -> BaseRetriever:
        """Get a retriever from the vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot get retriever.")
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})