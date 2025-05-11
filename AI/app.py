#!/usr/bin/env python
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"


def main():
    """Main entry point for the RAG-powered multi-agent assistant"""
    parser = argparse.ArgumentParser(description="RAG-Powered Multi-Agent Q&A Assistant.")
    parser.add_argument(
        "--ui",
        choices=["cli", "web"],
        default="web", 
        help="User interface type: 'cli' for command-line, 'web' for Streamlit web UI (default: web)."
    )
    
    ui_mode = "web"  # Default
    try:
        # Check if 'streamlit' is part of the execution command
        is_streamlit_run = any('streamlit' in arg.lower() for arg in sys.argv[:2])

        if is_streamlit_run:
            # If `streamlit run script.py -- --ui cli` is used, parse args after '--'
            if '--' in sys.argv:
                args_to_parse = sys.argv[sys.argv.index('--') + 1:]
                args = parser.parse_args(args_to_parse)
                ui_mode = args.ui
            else:
                # Default to 'web' if run with streamlit and no specific '--ui' arg after '--'
                ui_mode = "web" 
        else:
            # Standard parsing if script is run directly, e.g., `python script.py --ui cli`
            args = parser.parse_args()
            ui_mode = args.ui

    except SystemExit:  # Happens with --help
        logger.info("Argument parser exited (e.g., due to --help).")
        sys.exit(0) 
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}. Defaulting to 'web' UI.", exc_info=True)
        ui_mode = "web" 

    logger.info(f"Selected UI mode: {ui_mode}")

    if ui_mode == "web":
        logger.info("Starting Streamlit Web UI...")
        from src.ui.web import create_streamlit_ui
        create_streamlit_ui()
    elif ui_mode == "cli":
        logger.info("Starting Command-Line Interface (CLI)...")
        from src.ui.cli import create_cli
        create_cli()
    else: 
        logger.error(f"Invalid UI mode specified: {ui_mode}. Defaulting to web.")
        from src.ui.web import create_streamlit_ui
        create_streamlit_ui()

if __name__ == "__main__":
    main()