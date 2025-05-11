import logging
from typing import List, Dict, Any

# Import the assistant
from src.assistant import RAGMultiAgentAssistant
from src.agent.manager import WORDNET_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_cli():
    """Create a command-line interface for the assistant"""
    print("RAG-Powered Multi-Agent Q&A Assistant (CLI Mode)")
    print("Type 'exit' or 'quit' to stop.")
    print("-------------------------------------------------")
    
    if not WORDNET_AVAILABLE:
        print("WARNING: NLTK 'wordnet' corpus might not be available (check startup logs).")
        print("The dictionary tool ('define' command) may not function correctly.\n")

    assistant = None
    try:
        assistant = RAGMultiAgentAssistant()
        assistant.initialize() 
        print("Assistant initialized successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not initialize the assistant for CLI.")
        print(f"Error details: {e}")
        print("Please check logs, ensure `./docs` directory exists, models/API tokens are accessible, and all dependencies are installed.")
        if not WORDNET_AVAILABLE:
            print("Specifically, if 'wordnet' download failed, ensure internet and NLTK permissions, then restart.")
        return 

    while True:
        try:
            query = input("\nAsk a question: ").strip()
        except EOFError: 
            print("\nExiting...")
            break
        except KeyboardInterrupt: 
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not query:
            continue

        print("Processing your query...")
        result = assistant.answer_question(query)

        print("\n" + "=" * 60)
        print(f"You Asked: {result.get('query', query)}")
        print(f"Tool Used: {result.get('tool_used', 'N/A')}")
        if result.get('tool_input') and result.get('tool_used') not in ['knowledge_base', 'error_in_processing', 'initialization_error', 'system_component_error']:
            print(f"Input to Tool: {result.get('tool_input')}")
        print("-" * 60)
        print(f"Assistant Answered: {result.get('answer', 'No answer found or an error occurred.')}")
        print("=" * 60)

        if result.get('tool_used') == "knowledge_base" and result.get("success", False):
            retrieved_docs = assistant.get_retrieved_documents_for_last_query()
            if retrieved_docs:
                print("\nRetrieved context snippets for the answer:")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', 'Unknown Source')
                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    print(f"\n--- Snippet {i+1} (Source: {source}) ---")
                    print(content_preview)
            else:
                print("No specific context snippets were retrieved by the RAG tool for this answer.")