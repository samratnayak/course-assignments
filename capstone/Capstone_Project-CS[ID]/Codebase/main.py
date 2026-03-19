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
from typing import Dict, Optional

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
from embedding_model import EmbeddingModel
from skill_matcher import SkillMatcher
from ats_scorer import ATSScorer
from resume_optimizer import ResumeOptimizer
from cv_utils import display_cv_preview, get_user_feedback, apply_feedback_to_sections




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
        "input_files",
        type=str,
        nargs='+',
        help="Path(s) to input file(s) (PDF, DOCX, or TXT) containing unstructured user profile data. "
             "You can provide multiple files separated by spaces."
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
        help="Optional: Job description text directly, or path to job description file, or 'NA' to skip. "
             "If not provided, system will check input file and prompt user if needed."
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional OpenAI API key for GPT-4o optimization (can also be set via OPENAI_API_KEY environment variable)"
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
        
        # Set OpenAI API key if provided
        if args.api_key:
            config.openai_api_key = args.api_key
        elif os.getenv("OPENAI_API_KEY"):
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Step 1: Extract text from input document(s)
        print("\n[Step 1] Extracting text from input document(s)...")
        extractor = DocumentExtractor()
        
        # Process multiple input files
        input_files = args.input_files
        all_extracted_texts = []
        valid_files = []
        
        for input_file in input_files:
            # Check if file exists
            file_path = input_file
            if not os.path.exists(file_path):
                # Try relative to Codebase directory
                file_path = os.path.join(config.base_dir, input_file)
                if not os.path.exists(file_path):
                    print(f"⚠ Warning: Input file not found: {input_file} - skipping")
                    continue
            
            try:
                # Extract text from this file
                file_text = extractor.extract_text(file_path)
                all_extracted_texts.append(file_text)
                valid_files.append(file_path)
                print(f"  ✓ Extracted {len(file_text)} characters from: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ⚠ Error extracting from {os.path.basename(file_path)}: {e} - skipping")
                continue
        
        if not all_extracted_texts:
            raise FileNotFoundError(f"No valid input files found. Checked: {', '.join(input_files)}")
        
        # Combine all extracted texts
        extracted_text = "\n\n".join(all_extracted_texts)
        print(f"✓ Total extracted {len(extracted_text)} characters from {len(valid_files)} file(s)")
        
        # Step 2: Check for job description in input file(s)
        print("\n[Step 2] Checking for job description in input file(s)...")
        job_parser = JobDescriptionParser()  # Initialize without LLM first for detection
        profile_text, job_description_text = job_parser.separate_profile_and_job_description(extracted_text)
        
        if job_description_text:
            print("✓ Job description detected in input file(s)")
            extracted_text = profile_text  # Use only profile text for resume extraction
        else:
            print("✓ No job description found in input file(s)")
        
        # Step 3: Initialize LLM 1 (Fast Model) and LLM 2 (GPT-4o) for optimization
        print("\n[Step 3] Initializing LLM Models...")
        print("  - LLM 1 (Fast Extraction): Ollama (Mistral 7B) or Flan-T5-XL")
        print("  - LLM 2 (Optimization): GPT-4o (OpenAI) - Best cost-performance for resume optimization")
        cv_generator = CVGenerator(config)
        cv_generator.load_models()
        print(f"✓ LLM 1 (Extraction Model): {cv_generator.primary_model}")
        print(f"✓ LLM 2 (Optimization Model): {cv_generator.optimization_model}")
        
        # Check LangChain availability
        try:
            from langchain_parsers import is_langchain_available, get_parser_info
            if is_langchain_available():
                print(f"✓ {get_parser_info()}")
            else:
                print(f"ℹ {get_parser_info()}")
        except ImportError:
            pass
        
        # Update job parser with LLM model
        job_parser.llm_model = cv_generator
        
        # Step 4: LLM 1 → Extract Skills & Requirements
        print("\n[Step 4] LLM 1 → Extracting Skills & Requirements from Resume...")
        resume_extractor = ResumeExtractor(cv_generator)
        structured_data = resume_extractor.extract_resume_data(extracted_text)
        print("✓ Resume data extracted successfully")
        print(f"  - Name: {structured_data.get('name', 'N/A')}")
        print(f"  - Skills: {len(structured_data.get('skills', []))}")
        print(f"  - Tools: {len(structured_data.get('tools', []))}")
        print(f"  - Soft Skills: {len(structured_data.get('soft_skills', []))}")
        print(f"  - All Skills: {len(structured_data.get('all_skills', []))}")
        
        # Collect all skills from various sources
        resume_skills = []
        # Add skills from different categories
        resume_skills.extend(structured_data.get('skills', []))
        resume_skills.extend(structured_data.get('tools', []))
        resume_skills.extend(structured_data.get('all_skills', []))
        # Remove duplicates while preserving order
        seen = set()
        resume_skills = [s for s in resume_skills if s and not (s.lower() in seen or seen.add(s.lower()))]
        
        # Step 5: Handle job description for CV tailoring
        job_requirements = None
        job_text_to_parse = None
        
        # Priority 1: Use job description from input file if detected
        if job_description_text:
            job_text_to_parse = job_description_text
            print("\n[Step 5] Using job description from input file...")
        
        # Priority 2: Check command line argument
        elif args.job_description:
            if args.job_description.upper() == 'NA':
                print("\n[Step 5] Job description skipped (NA specified)")
                job_text_to_parse = None
            elif os.path.exists(args.job_description):
                # It's a file path
                print("\n[Step 5] Loading job description from specified file...")
                with open(args.job_description, 'r', encoding='utf-8') as f:
                    job_text_to_parse = f.read()
            else:
                # Treat as direct text input
                print("\n[Step 5] Using job description text from command line...")
                job_text_to_parse = args.job_description
        
        # Priority 3: Ask user interactively (before conversational mode)
        if job_text_to_parse is None and (not args.job_description or args.job_description.upper() != 'NA'):
            print("\n" + "=" * 80)
            print("JOB DESCRIPTION INPUT")
            print("=" * 80)
            print("No job description found. You can provide one for ATS-optimized CV tailoring.")
            print("Options:")
            print("  1. Enter job description text directly (paste the job description)")
            print("  2. Enter path to job description file")
            print("  3. Type 'NA' or press Enter to skip job description")
            print("=" * 80)
            
            user_input = input("\nEnter job description text or file path (or 'NA' to skip): ").strip()
            
            if user_input and user_input.upper() != 'NA':
                if os.path.exists(user_input):
                    # It's a file path
                    with open(user_input, 'r', encoding='utf-8') as f:
                        job_text_to_parse = f.read()
                    print(f"✓ Job description loaded from: {user_input}")
                else:
                    # Treat as direct text input
                    job_text_to_parse = user_input
                    print(f"✓ Job description text accepted ({len(user_input)} characters)")
            else:
                print("✓ Proceeding without job description")
        
        # Parse job description if available
        jd_skills = []
        jd_keywords = []
        if job_text_to_parse:
            print("\n[Step 5] LLM 1 → Extracting Skills & Requirements from Job Description...")
            job_requirements = job_parser.parse_job_description(job_text_to_parse, from_input_file=bool(job_description_text))
            print("✓ Job description parsed")
            jd_skills = job_requirements.get('skills_required', []) or job_requirements.get('keywords', [])
            jd_keywords = job_requirements.get('keywords', [])
            print(f"  - JD Skills extracted: {len(jd_skills)}")
            print(f"  - Keywords extracted: {len(jd_keywords)}")
        else:
            print("\n[Step 5] No job description provided. Skipping optimization pipeline...")
            job_requirements = None
        
        # Step 6: Initialize Embedding Model for Semantic Matching
        print("\n[Step 6] Initializing Embedding Model for Semantic Matching...")
        embedding_model = EmbeddingModel()
        
        # Step 7: Skill Gap Detection using Semantic Similarity
        missing_skills = []
        if jd_skills:
            print("\n[Step 7] Skill Gap Detection using Semantic Similarity...")
            skill_matcher = SkillMatcher(embedding_model)
            missing_skills = skill_matcher.find_missing_skills(resume_skills, jd_skills)
            print(f"✓ Skill gap analysis complete")
            print(f"  - Missing skills detected: {len(missing_skills)}")
            if missing_skills:
                print(f"  - Missing: {', '.join(missing_skills[:5])}{'...' if len(missing_skills) > 5 else ''}")
        else:
            print("\n[Step 7] Skipping skill gap detection (no job description)")
            skill_matcher = None
        
        # Step 8: Initialize ATS Scorer
        if jd_skills and skill_matcher:
            print("\n[Step 8] Initializing ATS Scorer...")
            ats_scorer = ATSScorer(embedding_model, skill_matcher)
            print("✓ ATS Scorer initialized")
        else:
            ats_scorer = None
        
        # Step 9: LLM 2 → Resume Optimization with Self-Improvement Loop
        if jd_skills and ats_scorer and skill_matcher:
            print("\n[Step 9] LLM 2 → Resume Optimization with Self-Improvement Loop...")
            resume_optimizer = ResumeOptimizer(cv_generator, ats_scorer, skill_matcher, target_score=0.8, max_iterations=3)
            
            # Combine resume text for optimization
            resume_text = extracted_text  # Use original extracted text
            
            # Generate initial CV sections first
            print("\n[Step 9a] Generating initial CV sections...")
            cv_sections = cv_generator.generate_tailored_cv(
                structured_data, job_requirements, missing_skills
            )
            
            # Run self-improvement loop
            print("\n[Step 9b] Running self-improvement optimization loop...")
            optimization_result = resume_optimizer.optimize_with_self_improvement(
                resume_text, structured_data, job_text_to_parse, jd_skills, jd_keywords, 
                job_requirements, cv_sections
            )
            
            print(f"\n✓ Optimization complete!")
            print(f"  - Final ATS Score: {optimization_result['final_score']:.3f}")
            print(f"  - Iterations: {optimization_result['iterations']}")
            print(f"  - Target Achieved: {'Yes' if optimization_result['target_achieved'] else 'No'}")
            
            # Use optimized CV sections if available, otherwise regenerate
            if 'optimized_cv_sections' in optimization_result:
                print("\n[Step 10] Using optimized CV sections from self-improvement loop...")
                cv_sections = optimization_result['optimized_cv_sections']
            else:
                print("\n[Step 10] Regenerating optimized CV sections...")
                cv_sections = cv_generator.generate_tailored_cv(
                    structured_data, job_requirements, missing_skills
                )
        else:
            # Generate CV without optimization (no job description)
            print("\n[Step 9] Generating CV sections (no optimization - no job description)...")
            cv_sections = cv_generator.generate_tailored_cv(structured_data, job_requirements)
            
            # Calculate ATS score even without job description (for informational purposes)
            if embedding_model:
                print("\n[Step 9b] Calculating baseline ATS metrics...")
                # Create a simple ATS scorer for baseline calculation
                baseline_skill_matcher = SkillMatcher(embedding_model)
                baseline_ats_scorer = ATSScorer(embedding_model, baseline_skill_matcher)
                
                # Use empty job description for baseline score
                resume_text = "\n\n".join([f"{k}\n{v}" for k, v in cv_sections.items()])
                baseline_score = baseline_ats_scorer.calculate_ats_score(
                    resume_text, resume_skills, "", [], []
                )
                print(f"  - Baseline ATS Score: {baseline_score['overall_score']:.3f}")
                print(f"    (Note: This is a baseline score. For accurate ATS scoring, provide a job description.)")
        
        # Step 11: Conversational CV refinement loop
        print("\n" + "=" * 80)
        print("CONVERSATIONAL CV REFINEMENT MODE")
        print("=" * 80)
        print("The system will now show you the CV and allow you to provide feedback")
        print("for iterative improvements. Type 'stop' when you're satisfied.")
        print("=" * 80)
        
        iteration = 1
        formatter = CVFormatter()
        
        # Validate and sanitize candidate name
        raw_name = structured_data.get('name', '').strip('"\'')
        # Remove trailing commas and quotes
        raw_name = raw_name.rstrip(',').strip().strip('"\'')
        
        # Validate the extracted name using the resume extractor's validation
        if not raw_name or not resume_extractor._is_valid_name(raw_name):
            print(f"⚠ Warning: Invalid name extracted ('{raw_name}'). Using 'Candidate' as default.")
            raw_name = 'Candidate'
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Sanitize candidate name for filename (remove quotes, commas, and other invalid chars)
            sanitized_name = raw_name.replace('"', '').replace("'", '').replace(',', '').replace(' ', '_')
            # Remove any remaining invalid filename characters
            sanitized_name = ''.join(c for c in sanitized_name if c.isalnum() or c in ('_', '-'))
            if not sanitized_name:  # If name becomes empty after sanitization
                sanitized_name = 'Candidate'
            output_filename = f"cv_{sanitized_name}.{args.format}"
            output_path = os.path.join(config.output_dir, output_filename)
        
        # Get clean candidate name for display (without quotes)
        candidate_name = raw_name
        
        # Step 11a: Save initial CV before feedback loop
        print("\n[Step 11a] Saving initial CV...")
        formatter.format_cv(
            cv_sections,
            output_path,
            format_type=args.format,
            candidate_name=candidate_name
        )
        print(f"✓ Initial CV saved to: {output_path}")
        
        while True:
            # Display current CV preview
            display_cv_preview(cv_sections, iteration)
            
            # Get user feedback (pass cv_generator for LLM-based parsing)
            feedback_dict = get_user_feedback(cv_sections, cv_generator)
            
            # Check if user wants to stop
            if '_stop' in feedback_dict:
                print("\n✓ Finalizing CV based on your feedback...")
                break
            
            # Check if there's any feedback to process
            if not feedback_dict:
                print("\n✓ No changes requested. Keeping current version.")
                # Ask if user wants to continue or stop
                continue_choice = input("\nWould you like to continue refining? (yes/no): ").strip().lower()
                if continue_choice in ['no', 'n', 'stop', 'done', 'exit', 'quit']:
                    break
                iteration += 1
                continue
            
            # Apply feedback and regenerate sections
            print("\n[Processing feedback and regenerating sections...]")
            section_feedback = apply_feedback_to_sections(cv_sections, feedback_dict, cv_generator)
            
            if section_feedback:
                cv_sections = cv_generator.regenerate_multiple_sections_with_feedback(
                    cv_sections, structured_data, section_feedback, job_requirements
                )
                print("✓ CV sections updated based on your feedback")
                
                # Save updated CV after regeneration
                print(f"\n[Saving updated CV (iteration {iteration + 1})...]")
                formatter.format_cv(
                    cv_sections,
                    output_path,
                    format_type=args.format,
                    candidate_name=candidate_name
                )
                print(f"✓ Updated CV saved to: {output_path}")
            else:
                print("⚠ No valid feedback to process")
            
            iteration += 1
        
        # Step 12: Save final CV (ensures latest version is saved)
        print("\n[Step 12] Saving final CV...")
        formatter.format_cv(
            cv_sections,
            output_path,
            format_type=args.format,
            candidate_name=candidate_name
        )
        print(f"✓ Final CV saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("CV GENERATION COMPLETE!")
        print("=" * 80)
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        print(f"Format: {args.format.upper()}")
        print(f"Iterations: {iteration}")
        
        # Display models used correctly
        llm1_model = "Mistral 7B (Ollama)" if cv_generator.primary_model == "ollama" else "Flan-T5-XL"
        llm2_model = "GPT-4o (OpenAI)" if cv_generator.optimization_model == "openai" else f"{cv_generator.optimization_model} (fallback)"
        print(f"LLM 1 (Extraction): {llm1_model}")
        print(f"LLM 2 (Optimization): {llm2_model}")
        
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
