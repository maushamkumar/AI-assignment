import os
import logging
from typing import Optional

# HuggingFace models
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

# Core LangChain
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.documents import Document
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceLLMManager:
    """Manages HuggingFace model interactions"""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> BaseLLM:
        """Initialize the LLM, trying local pipeline first, then HuggingFaceHub."""
        # Attempt 1: Local HuggingFacePipeline
        try:
            logger.info(f"Attempting to initialize local HuggingFacePipeline with model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            text_generation_pipeline = hf_pipeline(
                "text2text-generation", 
                model=model,
                tokenizer=tokenizer,
                max_length=512, 
                do_sample=False 
            )
            llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info(f"Successfully initialized local HuggingFacePipeline with model: {self.model_name}")
            return llm
        except Exception as e:
            logger.warning(f"Failed to load model '{self.model_name}' locally using HuggingFacePipeline: {e}. Trying HuggingFaceHub as fallback.")

        # Attempt 2: HuggingFaceHub (requires API token)
        try:
            logger.info(f"Attempting to initialize HuggingFaceHub with model: {self.model_name}")
            if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
                logger.error("HUGGINGFACEHUB_API_TOKEN environment variable not set. HuggingFaceHub will likely fail.")
                raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

            llm = HuggingFaceHub(
                repo_id=self.model_name,
                model_kwargs={"temperature": 0.1, "max_length": 512} 
            )
            logger.info(f"Successfully initialized HuggingFaceHub with model: {self.model_name}")
            return llm
        except Exception as hub_e:
            logger.error(f"Failed to initialize HuggingFaceHub model '{self.model_name}': {hub_e}", exc_info=True)
            raise ValueError(
                f"Could not initialize LLM. Local HuggingFacePipeline failed, and HuggingFaceHub also failed. "
                f"Last Hub error: {hub_e}"
            )

    def create_qa_chain(self, retriever: BaseRetriever) -> Runnable:
        """Create a RAG chain for question answering"""
        template = """
        Answer the question based ONLY on the following context.
        If the context does not contain the answer, state that you don't have enough information from the documents.
        Do not make up an answer or use external knowledge.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs(docs: List[Document]) -> str:
            if not docs:
                return "No context provided."
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser() # Ensure output is a string
        )
        logger.info("RAG QA chain created successfully.")
        return rag_chain