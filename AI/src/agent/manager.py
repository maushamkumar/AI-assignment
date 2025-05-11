import logging
import re
import math
from typing import List, Dict, Any, Tuple, Optional

# Core LangChain
from langchain_core.language_models import BaseLLM
from langchain_core.tools import Tool as LangChainTool
from langchain_core.runnables import Runnable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global NLTK setup
WORDNET_AVAILABLE = False
wordnet = None
try:
    import nltk
    logger.info("Attempting to download NLTK 'wordnet' corpus...")
    nltk.download('wordnet', quiet=False, raise_on_error=True)
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
    logger.info("NLTK 'wordnet' corpus loaded successfully. Dictionary tool should be available.")
except ImportError:
    logger.warning("NLTK library not found. Please install it (`pip install nltk`) for the dictionary tool to work.")
except Exception as e:
    logger.warning(f"NLTK 'wordnet' corpus not available or download failed: {e}. "
                   "Dictionary tool will be limited or unavailable. "
                   "Ensure you have an internet connection and NLTK can write to its data directory.")

class SimpleAgentManager:
    """Manages a simplified agent with tools"""

    def __init__(self, llm: 'BaseLLM', rag_chain: 'Runnable'):
        self.llm = llm
        self.rag_chain = rag_chain
        self.tools = self._create_tools()

    def _create_calculator_tool(self) -> 'LangChainTool':
        """Create a calculator tool"""
        def calculator(query: str) -> str:
            """Calculate the result of a mathematical expression."""
            try:
                processed_query = query.lower()
                processed_query = processed_query.replace('^', '**')
                processed_query = processed_query.replace('x', '*')

                if not re.search(r'\d', processed_query) and not any(c in processed_query for c in ['pi', 'e']):
                     if not re.search(r'sqrt\(|pow\(|log\(|sin\(|cos\(|tan\(', processed_query):
                        logger.warning(f"Calculator received potentially non-mathematical query: '{query}'")

                safe_dict = {
                    'abs': abs, 'round': round, 'min': min, 'max': max,
                    'pow': pow, 'sqrt': math.sqrt,
                    'pi': math.pi, 'e': math.e,
                    'log10': math.log10, 'log': math.log, 'ln': math.log,
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                }
                
                result = eval(processed_query, {"__builtins__": {}}, safe_dict)
                return f"The result of '{query}' is: {result}"
            except SyntaxError:
                logger.error(f"Calculator SyntaxError for query '{query}': Malformed expression.")
                return f"Error: The mathematical expression '{query}' is malformed or incomplete."
            except NameError as e:
                logger.error(f"Calculator NameError for query '{query}': {e}. Likely an unsupported variable or function.")
                return f"Error: '{query}' contains an unsupported variable or function. Only basic math and predefined functions are allowed."
            except TypeError as e:
                logger.error(f"Calculator TypeError for query '{query}': {e}. Check function arguments.")
                return f"Error: There was a type error in processing '{query}'. Ensure functions have correct arguments (e.g., sqrt(number))."
            except ZeroDivisionError:
                logger.error(f"Calculator ZeroDivisionError for query '{query}'.")
                return f"Error: Division by zero in expression '{query}'."
            except Exception as e:
                logger.error(f"Calculator error for query '{query}': {e}", exc_info=True)
                return f"Error calculating '{query}': An unexpected error occurred. Invalid expression or unsupported operation."

        return LangChainTool(
            name="calculator",
            description="Useful for mathematical calculations. Input should be a mathematical expression (e.g., '2 + 2', 'sqrt(16)', 'pi * 2').",
            func=calculator
        )

    def _create_dictionary_tool(self) -> 'LangChainTool':
        """Create a dictionary tool"""
        def define_word(word_query: str) -> str:
            """Get the definition of a word."""
            # Relies on global WORDNET_AVAILABLE and wordnet object
            if not WORDNET_AVAILABLE or wordnet is None:
                return "Dictionary tool is not available (NLTK 'wordnet' corpus not installed or failed to load properly at startup)."
            
            try:
                word_to_define = word_query.strip().lower()
                # More robust prefix stripping, handling cases like "define: word" or "define word?"
                prefixes_to_remove = ["define ", "definition of ", "meaning of ", "what is the meaning of ", "what is the definition of "]
                for prefix in prefixes_to_remove:
                    if word_to_define.startswith(prefix):
                        word_to_define = word_to_define[len(prefix):].strip()
                
                word_to_define = re.sub(r'[?:!.,;]$', '', word_to_define).strip() # Remove trailing punctuation more cleanly

                if not word_to_define:
                    return "Please provide a word for the dictionary (e.g., 'define serendipity' or 'meaning of ephemeral')."

                synsets = wordnet.synsets(word_to_define)
                
                if not synsets:
                    return f"No definition found for '{word_to_define}'."
                
                definitions = []
                for i, syn in enumerate(synsets[:3]): # Get top 3 synsets
                    defn = syn.definition()
                    examples = syn.examples()
                    pos = syn.pos() # Part of speech tag (e.g., 'n', 'v', 'a')
                    pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective satellite', 'r': 'adverb'}
                    readable_pos = pos_map.get(pos, pos) # Convert to readable form
                    
                    entry = f"({readable_pos}) {defn}"
                    if examples:
                        entry += f" (e.g., \"{examples[0]}\")" # Add first example if available
                    definitions.append(entry)
                
                return f"Definitions of '{word_to_define}':\n- " + "\n- ".join(definitions)
            except Exception as e:
                logger.error(f"Dictionary error for word '{word_query}': {e}", exc_info=True)
                return f"Error getting definition for '{word_query}': An unexpected error occurred."
        
        return LangChainTool(
            name="dictionary",
            description="Provides the definition of a word. Input should be a single word or a phrase like 'define X' or 'meaning of X'.",
            func=define_word
        )

    def _create_rag_tool(self) -> 'LangChainTool':
        """Create a RAG tool"""
        def rag_qa(query: str) -> str:
            try:
                # The RAG chain should output a string directly due to StrOutputParser
                result_text = self.rag_chain.invoke(query)
                return str(result_text) # Ensure it's a string
            except Exception as e:
                logger.error(f"Error in RAG tool with query '{query}': {e}", exc_info=True)
                return "I encountered an issue trying to find information in the documents for your query."
        
        return LangChainTool(
            name="knowledge_base",
            description="Provides answers from the knowledge base for general questions, information retrieval from documents, or when other tools are not suitable.",
            func=rag_qa
        )

    def _create_tools(self) -> List['LangChainTool']:
        """Create all tools"""
        tools = [
            self._create_calculator_tool(),
            self._create_dictionary_tool(),
            self._create_rag_tool()
        ]
        logger.info(f"Created tools: {[tool.name for tool in tools]}")
        return tools

    def decide_tool(self, query: str) -> Tuple[str, 'LangChainTool', str]:
        """
        Decide which tool to use based on the query and extract the relevant argument for the tool.
        Returns: (tool_name, tool_object, tool_input)
        """
        query_lower = query.lower().strip()
        
        # Priority 1: Calculator
        calc_keywords_strict = ['calculate', 'compute', 'evaluate', 'what is the value of', 'solve for']
        calc_keywords_soft = ['what is', 'math for']

        potential_expr_for_calc = ""
        has_calc_keyword_match = False

        for keyword in calc_keywords_strict:
            if query_lower.startswith(keyword + " "):
                potential_expr_for_calc = query[len(keyword):].strip()
                has_calc_keyword_match = True
                break
        
        if not has_calc_keyword_match:
            for keyword in calc_keywords_soft:
                if query_lower.startswith(keyword + " "):
                    temp_expr_candidate = query[len(keyword):].strip()
                    temp_expr_candidate_lower = temp_expr_candidate.lower()
                    if (re.search(r'\d', temp_expr_candidate_lower) and \
                        re.search(r'[+\-*/^().%]', temp_expr_candidate_lower)) or \
                       (temp_expr_candidate_lower in ['pi', 'e']) or \
                       (re.search(r'^\s*(sqrt|pow|log|ln|sin|cos|tan|abs|round|min|max)\s*\(.+\)', temp_expr_candidate_lower)):
                        potential_expr_for_calc = temp_expr_candidate
                        has_calc_keyword_match = True
                        break
        
        if has_calc_keyword_match and potential_expr_for_calc:
            potential_expr_for_calc = re.sub(r'^(the result of|the value of|of|the|for|to|me|is)\s+', '', potential_expr_for_calc, flags=re.IGNORECASE).strip()
            potential_expr_for_calc = potential_expr_for_calc.rstrip('?:!.,;')

            is_math_expression = (
                (re.search(r'\d', potential_expr_for_calc.lower()) and re.search(r'[+\-*/^().%]', potential_expr_for_calc.lower())) or
                (potential_expr_for_calc.lower() in ['pi', 'e']) or
                (re.search(r'^\s*(sqrt|pow|log|ln|sin|cos|tan|abs|round|min|max)\s*\(.+\)', potential_expr_for_calc.lower()))
            )
            if is_math_expression and potential_expr_for_calc:
                logger.info(f"Calculator selected. Passing input: '{potential_expr_for_calc}' (derived from query: '{query}')")
                return "calculator", next(t for t in self.tools if t.name == "calculator"), potential_expr_for_calc

        # Priority 2: Dictionary
        dict_keywords = ['define', 'definition of', 'meaning of', "what's the meaning of", "what is the definition of", 'dictionary for']
        tool_input_for_dict = ""

        for keyword in dict_keywords:
            # Ensure there's a space after the keyword to avoid partial matches like "defineable"
            if query_lower.startswith(keyword + " "): 
                tool_input_for_dict = query[len(keyword):].strip() 
                break
        
        # Handle cases like "define word" without a trailing space in the keyword list
        if not tool_input_for_dict and query_lower.startswith("define "):
             match = re.match(r'define\s+(.+)', query, flags=re.IGNORECASE)
             if match:
                tool_input_for_dict = match.group(1).strip()
        
        if tool_input_for_dict:
            tool_input_for_dict = tool_input_for_dict.rstrip('?:!.,;')
            if tool_input_for_dict: 
                logger.info(f"Dictionary selected. Passing input: '{tool_input_for_dict}' (derived from query: '{query}')")
                return "dictionary", next(t for t in self.tools if t.name == "dictionary"), tool_input_for_dict

        # Default to knowledge_base (RAG)
        logger.info(f"Defaulting to knowledge_base for query: '{query}'")
        return "knowledge_base", next(t for t in self.tools if t.name == "knowledge_base"), query
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query by selecting the appropriate tool"""
        try:
            tool_name, tool_obj, tool_input_str = self.decide_tool(query)
            
            logger.info(f"Processing query: '{query}'. Tool selected: '{tool_name}'. Input to tool: '{tool_input_str}'")
            
            if tool_obj and hasattr(tool_obj, 'func') and callable(tool_obj.func):
                answer = tool_obj.func(tool_input_str)
            else:
                logger.error(f"Tool '{tool_name}' is not configured correctly or has no callable function.")
                answer = f"Error: The tool '{tool_name}' could not be executed due to configuration issues."

            is_successful = not (isinstance(answer, str) and answer.lower().startswith("error:"))

            return {
                "query": query,
                "answer": answer,
                "tool_used": tool_name,
                "tool_input": tool_input_str,
                "success": is_successful
            }
        except Exception as e:
            logger.error(f"Error processing query '{query}' in SimpleAgentManager: {e}", exc_info=True)
            return {
                "query": query,
                "answer": f"An unexpected error occurred while processing your query: {str(e)}",
                "tool_used": "error_in_processing",
                "tool_input": query,
                "success": False
            }