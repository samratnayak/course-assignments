"""
CV Creation using LLMs - Capstone Project (CS[01])

This program uses Large Language Models (LLMs) to create professional CVs/Resumes
from unstructured user profile data or documents.

Input: Sample unstructured user profile data (text file, PDF, or Word document)
Output: Final CVs in DOCX/PDF format

Uses at least 2 LLMs:
1. Gemma 3 1B via Ollama (primary, if available)
2. Flan-T5-XL (secondary/backup)

Main entry point for the CV creation application.
"""

import sys
import os
import argparse
import torch
import transformers
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

# Add current directory to path for imports (all files in same directory per instructions)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from document_extractor import DocumentExtractor
from resume_extractor import ResumeExtractor
from job_parser import JobDescriptionParser
from cv_generator import CVGenerator
from cv_formatter import CVFormatter


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="CV Creation using LLMs - Generate professional CVs from unstructured data"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file (PDF, DOCX, or TXT) containing unstructured user profile data"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: output/cv_output.docx in Codebase directory)"
    )
    
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["docx", "txt", "pdf"],
        default="docx",
        help="Output format: docx, txt, or pdf (default: docx)"
    )
    
    parser.add_argument(
        "-j", "--job-description",
        type=str,
        default=None,
        help="Optional: Path to job description file for ATS-optimized CV tailoring"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key (can also be set via environment variable)"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the CV creation application.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize configuration
        print("=" * 80)
        print("CV Creation using LLMs - Capstone Project")
        print("=" * 80)
        config = Config()
        
        # Check if input file exists
        input_file = args.input_file
        if not os.path.exists(input_file):
            # Try relative to Codebase directory
            input_file = os.path.join(config.base_dir, args.input_file)
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Step 1: Extract text from input document
        print("\n[Step 1] Extracting text from input document...")
        extractor = DocumentExtractor()
        extracted_text = extractor.extract_text(input_file)
        print(f"✓ Extracted {len(extracted_text)} characters from input document")
        
        # Step 2: Initialize CV Generator with multiple LLMs
        print("\n[Step 2] Initializing CV Generator with LLMs...")
        print("  - Loading Ollama (Gemma 3 1B)...")
        print("  - Loading Flan-T5-XL...")
        cv_generator = CVGenerator(config)
        cv_generator.load_models()
        print(f"✓ Primary model: {cv_generator.primary_model}")
        
        # Step 3: Extract structured resume data using LLM
        print("\n[Step 3] Extracting structured resume data using LLM...")
        resume_extractor = ResumeExtractor(cv_generator)
        structured_data = resume_extractor.extract_resume_data(extracted_text)
        print("✓ Resume data extracted successfully")
        print(f"  - Name: {structured_data.get('name', 'N/A')}")
        print(f"  - Education entries: {len(structured_data.get('education', []))}")
        print(f"  - Experience entries: {len(structured_data.get('experience', []))}")
        print(f"  - Skills: {len(structured_data.get('skills', []))}")
        
        # Step 4: Parse job description (if provided) for CV tailoring
        job_requirements = None
        if args.job_description:
            print("\n[Step 4] Parsing job description for CV tailoring...")
            job_parser = JobDescriptionParser(cv_generator)
            if os.path.exists(args.job_description):
                with open(args.job_description, 'r', encoding='utf-8') as f:
                    job_text = f.read()
                job_requirements = job_parser.parse_job_description(job_text)
                print("✓ Job description parsed")
                print(f"  - Keywords extracted: {len(job_requirements.get('keywords', []))}")
            else:
                print(f"⚠ Job description file not found: {args.job_description}")
        
        # Step 5: Generate tailored CV using LLMs
        print("\n[Step 5] Generating tailored, ATS-friendly CV using LLMs...")
        cv_sections = cv_generator.generate_tailored_cv(structured_data, job_requirements)
        print("✓ CV sections generated successfully")
        
        # Step 6: Format and save CV
        print("\n[Step 6] Formatting and saving CV...")
        formatter = CVFormatter()
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            candidate_name = structured_data.get('name', 'Candidate').replace(' ', '_')
            output_filename = f"cv_{candidate_name}.{args.format}"
            output_path = os.path.join(config.output_dir, output_filename)
        
        # Format CV
        candidate_name = structured_data.get('name', 'Candidate')
        formatter.format_cv(
            cv_sections,
            output_path,
            format_type=args.format,
            candidate_name=candidate_name
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("CV GENERATION COMPLETE!")
        print("=" * 80)
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        print(f"Format: {args.format.upper()}")
        print(f"Models used: {cv_generator.primary_model} (primary), Flan-T5-XL (secondary)")
        if job_requirements:
            print("CV tailored for job requirements: Yes")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
