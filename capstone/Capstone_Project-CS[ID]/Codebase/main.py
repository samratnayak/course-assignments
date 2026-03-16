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
from typing import Dict

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


def display_cv_preview(cv_sections: Dict[str, str], iteration: int = 1):
    """
    Display a preview of the CV sections for user review.
    
    Args:
        cv_sections: Dictionary of CV sections
        iteration: Current iteration number
    """
    print("\n" + "=" * 80)
    print(f"CV PREVIEW (Iteration {iteration})")
    print("=" * 80)
    
    for section_name, section_content in cv_sections.items():
        print(f"\n[{section_name}]")
        print("-" * 80)
        # Display first 500 characters of each section to keep it manageable
        preview = section_content[:500] if len(section_content) > 500 else section_content
        print(preview)
        if len(section_content) > 500:
            print("... (truncated)")
    
    print("\n" + "=" * 80)


def get_user_feedback() -> Dict[str, str]:
    """
    Get feedback from user for CV sections.
    
    Returns:
        Dictionary mapping section names to feedback strings, or empty dict if no feedback
    """
    print("\n" + "=" * 80)
    print("FEEDBACK OPTIONS")
    print("=" * 80)
    print("You can provide feedback in the following ways:")
    print("1. Type 'stop', 'done', 'exit', or 'quit' to finish and save the CV")
    print("2. Type 'all: [your feedback]' to provide feedback for all sections")
    print("3. Type '[section name]: [your feedback]' to provide feedback for a specific section")
    print("4. Type 'skip' or press Enter to keep current version and continue")
    print("=" * 80)
    
    user_input = input("\nEnter your feedback (or 'stop' to finish): ").strip()
    
    if not user_input or user_input.lower() in ['skip', '']:
        return {}
    
    if user_input.lower() in ['stop', 'done', 'exit', 'quit', 'finish']:
        return {'_stop': True}
    
    feedback_dict = {}
    
    # Check if feedback is for all sections
    if user_input.lower().startswith('all:'):
        feedback_text = user_input[4:].strip()
        if feedback_text:
            # Apply to all sections
            feedback_dict['_all'] = feedback_text
    else:
        # Try to parse section-specific feedback
        # Look for pattern: "Section Name: feedback"
        parts = user_input.split(':', 1)
        if len(parts) == 2:
            section_name = parts[0].strip()
            feedback_text = parts[1].strip()
            if feedback_text:
                feedback_dict[section_name] = feedback_text
        else:
            # If no section specified, treat as general feedback for all sections
            feedback_dict['_all'] = user_input
    
    return feedback_dict


def apply_feedback_to_sections(cv_sections: Dict[str, str], feedback_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Apply feedback to CV sections.
    
    Args:
        cv_sections: Current CV sections
        feedback_dict: Dictionary of feedback
        
    Returns:
        Dictionary mapping section names to feedback strings
    """
    section_feedback = {}
    
    # Handle general feedback for all sections
    if '_all' in feedback_dict:
        general_feedback = feedback_dict['_all']
        for section_name in cv_sections.keys():
            section_feedback[section_name] = general_feedback
    
    # Handle section-specific feedback
    for section_name, feedback in feedback_dict.items():
        if section_name != '_all' and section_name != '_stop':
            # Try to match section name (case-insensitive, partial match)
            matched = False
            for existing_section in cv_sections.keys():
                if section_name.lower() in existing_section.lower() or existing_section.lower() in section_name.lower():
                    section_feedback[existing_section] = feedback
                    matched = True
                    break
            
            if not matched:
                # If no match found, add as-is (might be a new section or typo)
                section_feedback[section_name] = feedback
    
    return section_feedback


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
        help="Optional: Path to job description file, or 'NA' to skip job description. "
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
        
        # Step 2: Check for job description in input file
        print("\n[Step 2] Checking for job description in input file...")
        job_parser = JobDescriptionParser()  # Initialize without LLM first for detection
        profile_text, job_description_text = job_parser.separate_profile_and_job_description(extracted_text)
        
        if job_description_text:
            print("✓ Job description detected in input file")
            extracted_text = profile_text  # Use only profile text for resume extraction
        else:
            print("✓ No job description found in input file")
        
        # Step 3: Initialize LLM 1 (Fast Model) and LLM 2 (GPT-4o) for optimization
        print("\n[Step 3] Initializing LLM Models...")
        print("  - LLM 1 (Fast Extraction): Ollama (Mistral 7B) or Flan-T5-XL")
        print("  - LLM 2 (Optimization): GPT-4o (OpenAI) - Best cost-performance for resume optimization")
        cv_generator = CVGenerator(config)
        cv_generator.load_models()
        print(f"✓ LLM 1 (Extraction Model): {cv_generator.primary_model}")
        print(f"✓ LLM 2 (Optimization Model): {cv_generator.optimization_model}")
        
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
        
        resume_skills = structured_data.get('all_skills', []) + structured_data.get('skills', [])
        
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
                print("\n[Step 5] Loading job description from specified file...")
                with open(args.job_description, 'r', encoding='utf-8') as f:
                    job_text_to_parse = f.read()
            else:
                print(f"\n[Step 5] ⚠ Job description file not found: {args.job_description}")
                job_text_to_parse = None
        
        # Priority 3: Ask user interactively (before conversational mode)
        if job_text_to_parse is None and (not args.job_description or args.job_description.upper() != 'NA'):
            print("\n" + "=" * 80)
            print("JOB DESCRIPTION INPUT")
            print("=" * 80)
            print("No job description found. You can provide one for ATS-optimized CV tailoring.")
            print("Options:")
            print("  1. Enter path to job description file")
            print("  2. Type 'NA' or press Enter to skip job description")
            print("=" * 80)
            
            user_input = input("\nEnter job description file path (or 'NA' to skip): ").strip()
            
            if user_input and user_input.upper() != 'NA':
                if os.path.exists(user_input):
                    with open(user_input, 'r', encoding='utf-8') as f:
                        job_text_to_parse = f.read()
                    print(f"✓ Job description loaded from: {user_input}")
                else:
                    print(f"⚠ File not found: {user_input}. Proceeding without job description.")
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
            resume_optimizer = ResumeOptimizer(cv_generator, ats_scorer, skill_matcher, target_score=0.8, max_iterations=5)
            
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
            
            # Regenerate CV sections with optimized content
            print("\n[Step 10] Regenerating optimized CV sections...")
            cv_sections = cv_generator.generate_tailored_cv(
                structured_data, job_requirements, missing_skills
            )
        else:
            # Generate CV without optimization (no job description)
            print("\n[Step 9] Generating CV sections (no optimization - no job description)...")
            cv_sections = cv_generator.generate_tailored_cv(structured_data, job_requirements)
        
        # Step 11: Conversational CV refinement loop
        print("\n" + "=" * 80)
        print("CONVERSATIONAL CV REFINEMENT MODE")
        print("=" * 80)
        print("The system will now show you the CV and allow you to provide feedback")
        print("for iterative improvements. Type 'stop' when you're satisfied.")
        print("=" * 80)
        
        iteration = 1
        formatter = CVFormatter()
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            candidate_name = structured_data.get('name', 'Candidate').replace(' ', '_')
            output_filename = f"cv_{candidate_name}.{args.format}"
            output_path = os.path.join(config.output_dir, output_filename)
        
        candidate_name = structured_data.get('name', 'Candidate')
        
        while True:
            # Display current CV preview
            display_cv_preview(cv_sections, iteration)
            
            # Get user feedback
            feedback_dict = get_user_feedback()
            
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
            section_feedback = apply_feedback_to_sections(cv_sections, feedback_dict)
            
            if section_feedback:
                cv_sections = cv_generator.regenerate_multiple_sections_with_feedback(
                    cv_sections, structured_data, section_feedback, job_requirements
                )
                print("✓ CV sections updated based on your feedback")
            else:
                print("⚠ No valid feedback to process")
            
            iteration += 1
        
        # Step 12: Format and save final CV
        print("\n[Step 12] Formatting and saving final CV...")
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
        print(f"Iterations: {iteration}")
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
