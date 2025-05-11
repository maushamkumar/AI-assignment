import streamlit as st
import logging
from typing import List, Dict, Any

# Import the assistant
from src.assistant import RAGMultiAgentAssistant
from src.agent.manager import WORDNET_AVAILABLE, wordnet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_streamlit_ui():
    """Create a Streamlit UI for the assistant"""
    st.set_page_config(page_title="RAG Multi-Agent Assistant", layout="wide", initial_sidebar_state="expanded")

    st.title("🤖 RAG-Powered Multi-Agent Q&A Assistant")
    st.markdown(
        "Ask questions about your documents (place them in the `./docs` folder), "
        "perform calculations (e.g., `calculate 2*pi*5`), or get word definitions (e.g., `define serendipity`)."
    )
    
    # Display a note about NLTK WordNet if it wasn't available at startup
    if not WORDNET_AVAILABLE:
        st.warning(
            "Dictionary tool (for 'define' queries) may be unavailable. "
            "The NLTK 'wordnet' corpus might not have downloaded correctly at startup. "
            "Please check the console logs. If it failed, ensure internet access, "
            "NLTK's ability to write to its data path, and restart the application."
        )

    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
        st.session_state.initialization_error_message = None 

    if st.session_state.assistant is None and st.session_state.initialization_error_message is None:
        with st.spinner("Initializing Assistant... This may take a moment (especially on first run or with large models)."):
            try:
                assistant_instance = RAGMultiAgentAssistant()
                assistant_instance.initialize() 
                st.session_state.assistant = assistant_instance
                st.success("Assistant initialized successfully!")
            except Exception as e:
                error_msg = (
                    f"Fatal Error Initializing Assistant: {e}. "
                    "Please check the console logs for more details. "
                    "Common issues: Ensure your `docs` directory exists (even if empty), "
                    "HuggingFace models are accessible (or `HUGGINGFACEHUB_API_TOKEN` is set for fallback), "
                    "and required packages (like `faiss-cpu` or `faiss-gpu`) are installed. "
                    "If the dictionary tool is unavailable, ensure NLTK 'wordnet' downloaded successfully (see logs)."
                )
                st.session_state.initialization_error_message = error_msg
                logger.critical(f"Streamlit UI: {error_msg}", exc_info=True) 
    
    if st.session_state.initialization_error_message:
        st.error(st.session_state.initialization_error_message)
        st.warning("The assistant could not be initialized. Please resolve the errors and refresh the page or restart the application.")
        if st.button("Retry Initialization"):
            st.session_state.assistant = None 
            st.session_state.initialization_error_message = None 
            st.rerun()
        return 

    assistant = st.session_state.assistant
    if not assistant: 
        st.error("Assistant is not available. This is an unexpected state. Please refresh or check logs.")
        if st.button("Attempt Re-Initialization"): # Added a button for this specific error case
            st.session_state.assistant = None 
            st.session_state.initialization_error_message = None 
            st.rerun()
        return

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Use columns for the input form elements
    col1, col2 = st.columns([3,1]) # Input field takes 3/4, clear button 1/4 of width

    with col1:
        with st.form(key="query_form"):
            query = st.text_input(
                "Ask your question:",
                key="query_input_box", 
                help="Examples: 'What is the main product of Contoso Ltd?', 'calculate (sqrt(64) + 2)^2', 'define ubiquitous'"
            )
            submit_button = st.form_submit_button("Submit Question")

    with col2:
        # "Clear History" button needs to be outside the form to not trigger form submission
        # Add some vertical space to align better with the form's submit button if needed.
        st.markdown("<br>", unsafe_allow_html=True) # Adjust spacing as needed
        if st.button("Clear History", key="clear_history_button"):
            st.session_state.query_history = []
            logger.info("Streamlit UI: Query history cleared.")
            st.rerun()

    if submit_button and query: # This check is now only for the form's submit button
        with st.spinner("🧠 Thinking..."):
            logger.info(f"Streamlit UI: Processing query: '{query}'")
            result = assistant.answer_question(query)
            st.session_state.query_history.insert(0, {"query": query, "result": result})
            logger.info(f"Streamlit UI: Result for '{query}': {result}")
        st.rerun()

    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("Interaction Log")
        for i, entry in enumerate(st.session_state.query_history):
            q = entry["query"]
            res = entry["result"]
            
            with st.container(): 
                st.markdown(f"**You Asked ({len(st.session_state.query_history) - i}):** {q}")

                if res and res.get("success"):
                    st.markdown(f"**🤖 Assistant Answered:**")
                    st.info(res.get("answer", "No specific answer provided."))

                    with st.expander("View Details & Context", expanded=False):
                        st.write(f"**Tool Used:** `{res.get('tool_used', 'N/A')}`")
                        if res.get('tool_input') and res.get('tool_used') not in ['knowledge_base', 'error_in_processing', 'initialization_error', 'system_component_error']:
                             st.write(f"**Input to Tool:** `{res.get('tool_input')}`")

                        if res.get("tool_used") == "knowledge_base":
                            # Retrieve documents using the assistant's method that stores them
                            retrieved_docs = assistant.get_retrieved_documents_for_last_query()
                            if retrieved_docs:
                                st.markdown("**Retrieved Context Snippets Used for Answer:**")
                                for doc_idx, doc in enumerate(retrieved_docs):
                                    source = doc.metadata.get('source', 'Unknown Source')
                                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                    st.markdown(f"**Snippet {doc_idx+1} (Source: {source}):**")
                                    st.caption(content_preview)
                                    st.markdown("---")
                            else:
                                st.write("No specific context snippets were retrieved by the RAG tool for this answer, or context retrieval failed.")
                elif res: 
                    st.error(f"Assistant Error: {res.get('answer', 'An unknown error occurred.')}")
                    with st.expander("Error Details", expanded=False):
                        st.write(f"**Tool Attempted:** `{res.get('tool_used', 'N/A')}`")
                        if res.get('tool_input'):
                            st.write(f"**Input to Tool:** `{res.get('tool_input')}`")
                else:
                    st.warning("No result was generated for this query, or an unexpected issue occurred.")
                st.markdown("---") 

    with st.sidebar:
        st.header("Controls & Info")
        if st.button("Force Reload Documents & Re-initialize"):
            with st.spinner("Force reloading and re-initializing... This may take some time."):
                try:
                    # Ensure we have an assistant instance to call initialize on,
                    # or create a new one if it was None (e.g. initial error)
                    if st.session_state.assistant is None:
                        logger.info("Force Reload: Assistant was None, creating new instance for re-initialization.")
                        st.session_state.assistant = RAGMultiAgentAssistant()
                        st.session_state.initialization_error_message = None # Clear any prior init error

                    st.session_state.assistant.initialize(force_reload=True)
                    st.success("Assistant re-initialized with fresh documents!")
                    # Clear history as context might have changed significantly
                    st.session_state.query_history = [] 
                except Exception as e:
                    st.session_state.initialization_error_message = f"Error during force reload: {e}"
                    logger.error(f"Streamlit UI: {st.session_state.initialization_error_message}", exc_info=True)
                    st.error(st.session_state.initialization_error_message)
                    st.session_state.assistant = None # Mark assistant as potentially broken
            st.rerun()

        st.markdown("---")
        st.markdown("Created by Mausham (Assignment Submission).")
        st.markdown(f"NLTK WordNet Available: {'Yes' if WORDNET_AVAILABLE and wordnet else 'No'}")