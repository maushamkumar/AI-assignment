# RAG-Powered Multi-Agent Q&A Assistant

This is a modular implementation of a RAG (Retrieval Augmented Generation) powered assistant that can answer questions based on document knowledge, perform calculations, and provide word definitions.

## Features

- **Document Processing**: Load and process documents from various formats (TXT, PDF, CSV, Markdown)
- **Vector Store Management**: Create, save, and load vector embeddings for efficient document retrieval
- **LLM Integration**: Use HuggingFace models locally or via API for text generation
- **Multi-Agent System**: Specialized tools for different query types:
  - RAG-based document question answering
  - Mathematical calculations
  - Word definitions (using NLTK WordNet)
- **Multiple UI Options**: Web interface (Streamlit) and CLI

## Project Structure

```
.
├── app.py                  # Main entry point
├── docs/                   # Place your documents here
├── .env                    # Environment variables (create from .env.template)
├── requirements.txt        # Project dependencies
├── src/                    # Source code
│   ├── agent/              # Agent components
│   │   └── manager.py      # Tool management and query processing
│   ├── document_processor/ # Document handling
│   │   └── processor.py    # Document loading and chunking
│   ├── llm/                # Language model components
│   │   └── manager.py      # LLM initialization and chain creation
│   ├── ui/                 # User interfaces
│   │   ├── cli.py          # Command-line interface
│   │   └── web.py          # Streamlit web interface
│   ├── vector_store/       # Vector embedding components
│   │   └── manager.py      # Vector store creation and retrieval
│   └── assistant.py        # Main assistant class integrating all components
└── vector_store/           # Generated vector store (will be created automatically)
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and add your HuggingFace API token if needed

## Usage

### Web Interface

Run the Streamlit web interface:

```bash
streamlit run app.py
```

or

```bash
python app.py --ui web
```

### Command Line Interface

Run the CLI version:

```bash
python app.py --ui cli
```

## Adding Documents

Place your documents in the `docs` directory. The system supports:
- `.txt` - Text files
- `.pdf` - PDF documents
- `.csv` - CSV files
- `.md` - Markdown files

The system will automatically load, process, and index these documents when initialized.

## Environment Variables

Configure the following in your `.env` file:

- `HUGGINGFACEHUB_API_TOKEN`: Your HuggingFace API token (required for API access)

## License

This project is for educational purposes.