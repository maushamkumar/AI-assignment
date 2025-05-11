import logging
from pathlib import Path
from typing import List

# Document processing
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document loading
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, docs_dir: str = "./docs"):
        self.docs_dir = Path(docs_dir) # Use Path object for consistency
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len, # Explicitly state length function
            is_separator_regex=False, # Default, but good to be explicit
        )

        # Map file extensions to appropriate loaders
        self.extension_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".md": UnstructuredMarkdownLoader
        }

    def load_documents(self) -> List[Document]:
        """Load all documents from the docs directory"""
        documents = []
        loaded_files = []

        # Create docs directory if it doesn't exist
        self.docs_dir.mkdir(parents=True, exist_ok=True) # Use Path object's mkdir

        # Load all files from docs directory
        for file_path in self.docs_dir.glob("*"):
            if file_path.is_file(): # Ensure it's a file
                ext = file_path.suffix.lower()
                if ext in self.extension_map:
                    try:
                        loader_class = self.extension_map[ext]
                        # Specify encoding for text-based files
                        if ext in [".txt", ".csv", ".md"]:
                            loader = loader_class(str(file_path), encoding='utf-8')
                        else:
                            loader = loader_class(str(file_path))

                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                        loaded_files.append(file_path.name)
                        logger.info(f"Loaded {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path.name}: {e}", exc_info=True)

        logger.info(f"Loaded {len(documents)} documents from {len(loaded_files)} files.")
        if not documents and not loaded_files:
            logger.warning(f"No documents found in directory: {self.docs_dir}")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            logger.warning("No documents provided to process_documents function.")
            return []
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")
        return chunks