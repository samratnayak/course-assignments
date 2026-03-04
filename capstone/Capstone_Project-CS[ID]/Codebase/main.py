"""
CV Creation using LLMs - Capstone Project

This program uses Large Language Models (LLMs) to create professional CVs/Resumes
based on user input information.

Main entry point for the CV creation application.
"""

import sys
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_generator import CVGenerator
from config import Config

def main():
    """
    Main function to run the CV creation application.
    """
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize CV Generator
        print("Initializing CV Generator...")
        cv_generator = CVGenerator(config)
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        cv_generator.load_model()
        
        # Example usage - this will be replaced with actual input handling
        print("\nCV Creation System Ready!")
        print("=" * 80)
        
        # TODO: Add input handling based on project requirements
        # This is a placeholder structure
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
