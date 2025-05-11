import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Core components
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.faiss import FAISS

# Import local modules
from src.document_processor.processor import DocumentProcessor
from src.vector_store.manager import VectorStoreManager
from src.llm.manager import HuggingFaceLLMManager
from src.agent.manager import SimpleAgentManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGMultiAgentAssistant:
    """Main class for the RAG-powered multi-agent assistant"""

    def __init__(self, docs_dir: str = "./docs", vector_store_path: str = "./vector_store"):
        self.docs_dir = docs_dir
        self.vector_store_path = vector_store_path

        try:
            self.document_processor = DocumentProcessor(docs_dir)
            self.vector_store_manager = VectorStoreManager() 
            self.llm_manager = HuggingFaceLLMManager() 
        except Exception as e:
            logger.critical(f"Fatal error during initialization of core components: {e}", exc_info=True)
            raise RuntimeError(f"Core component initialization failed: {e}")

        self.vector_store: Optional[FAISS] = None
        self.rag_chain: Optional[Runnable] = None
        self.agent_manager: Optional[SimpleAgentManager] = None
        self.retriever: Optional[BaseRetriever] = None

        self.is_initialized = False
        self.retrieved_docs_for_last_query: List[Document] = []

    def initialize(self, force_reload: bool = False):
        """Initialize the assistant. Can raise exceptions if components fail."""
        if self.is_initialized and not force_reload:
            logger.info("Assistant already initialized. Skipping re-initialization.")
            return

        logger.info(f"Initializing RAGMultiAgentAssistant... Force reload: {force_reload}")
        try:
            # 1. Load or Create Vector Store
            if not force_reload:
                logger.info(f"Attempting to load existing vector store from: {self.vector_store_path}")
                # load_vector_store sets self.vector_store_manager.vector_store
                self.vector_store = self.vector_store_manager.load_vector_store(self.vector_store_path)
            else:
                logger.info("Force reload: Invalidating any existing vector store instance from manager.")
                self.vector_store_manager.vector_store = None # Clear it in manager
                self.vector_store = None # Clear local reference

            if self.vector_store is None: # If not loaded (force_reload=True or load failed/not_exists)
                if force_reload:
                    logger.info("Force reload: Rebuilding vector store from documents.")
                else:
                    logger.info("No existing vector store found or load failed. Creating a new one.")

                documents = self.document_processor.load_documents()
                if not documents:
                    logger.warning(
                        "No documents found in docs directory. RAG tool might not be effective. "
                        "Creating a placeholder document for vector store initialization."
                    )
                    sample_doc = Document(
                        page_content="This is a placeholder document. Please add your actual documents to the 'docs' directory for the RAG system to work effectively.",
                        metadata={"source": "placeholder_system_doc"}
                    )
                    documents = [sample_doc]

                chunks = self.document_processor.process_documents(documents)
                if not chunks:
                    logger.error("No chunks created from documents. Vector store creation will fail.")
                    # Create a dummy chunk if processing somehow returns empty with placeholder
                    placeholder_chunk = Document(page_content="Placeholder chunk.", metadata={"source":"placeholder_system_chunk"})
                    chunks = [placeholder_chunk]
                    logger.warning("Created a dummy chunk to proceed with vector store initialization.")
                
                # create_vector_store sets self.vector_store_manager.vector_store
                self.vector_store = self.vector_store_manager.create_vector_store(chunks)
                self.vector_store_manager.save_vector_store(self.vector_store_path)
            
            # Ensure self.vector_store is synchronized with the manager's instance
            self.vector_store = self.vector_store_manager.vector_store 

            if not self.vector_store:
                 logger.error("Vector store is None even after attempting creation/loading. This is critical.")
                 raise ValueError("Failed to initialize or create vector store. It remains None.")

            # 2. Get Retriever (always re-created as it depends on vector_store)
            logger.info("Re-creating retriever...")
            self.retriever = self.vector_store_manager.get_retriever()
            if not self.retriever: 
                raise ValueError("Failed to initialize retriever from vector store.")

            # 3. Create RAG Chain (always re-created as it depends on retriever and LLM)
            logger.info("Re-creating RAG QA chain...")
            self.rag_chain = self.llm_manager.create_qa_chain(self.retriever)
            if not self.rag_chain:
                raise ValueError("Failed to create RAG QA chain.")

            # 4. Create Agent Manager (always re-created as it depends on LLM and RAG chain)
            logger.info("Re-creating SimpleAgentManager...")
            self.agent_manager = SimpleAgentManager(self.llm_manager.llm, self.rag_chain)
            if not self.agent_manager:
                raise ValueError("Failed to create SimpleAgentManager.")

            self.is_initialized = True
            logger.info("RAGMultiAgentAssistant initialized successfully.")

        except Exception as e:
            logger.critical(f"Error during RAGMultiAgentAssistant initialization: {e}", exc_info=True)
            self.is_initialized = False 
            raise 

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Answer a question using the agent"""
        if not self.is_initialized:
            logger.warning("Assistant not initialized. Attempting to initialize now (on-demand).")
            try:
                self.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize assistant on demand: {e}", exc_info=True)
                return {
                    "query": query,
                    "answer": f"System error: Assistant could not be initialized. Details: {e}",
                    "tool_used": "initialization_error",
                    "tool_input": query,
                    "success": False
                }

        if not self.agent_manager or not self.retriever:
             logger.error("Agent manager or retriever is not available even after initialization attempt.")
             return {
                "query": query,
                "answer": "Critical system error: Agent manager or retriever component is missing.",
                "tool_used": "system_component_error",
                "tool_input": query,
                "success": False
            }

        result = self.agent_manager.process_query(query)
        self.retrieved_docs_for_last_query = [] 
        if result.get("tool_used") == "knowledge_base" and result.get("success", False):
            try:
                # Use the original query for retrieving documents, not tool_input if it was modified
                self.retrieved_docs_for_last_query = self.retriever.invoke(query) 
            except Exception as e:
                logger.error(f"Error retrieving documents for display after RAG call: {e}", exc_info=True)
        return result

    def get_retrieved_documents_for_last_query(self) -> List[Document]:
        """Get the retrieved documents from the last RAG query"""
        return self.retrieved_docs_for_last_query