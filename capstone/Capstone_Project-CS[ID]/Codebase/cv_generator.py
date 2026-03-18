"""
CV Generator Module

This module contains the CVGenerator class that handles CV creation using multiple LLMs.

Pipeline Architecture:
1. LLM 1 (Fast Model - Mistral 7B/Flan-T5-XL): Extract Skills & Requirements
2. Embedding Model (all-MiniLM-L6-v2): Semantic Matching
3. Skill Gap Detection: Compare resume vs job skills
4. LLM 2 (Stronger Model - Same as LLM 1 but with optimization prompts): Resume Optimization
5. ATS Scoring: Calculate compatibility score
6. Self-Improvement Loop: Iteratively improve until target score

Model Selection Justification (Research-Based):

1. Mistral 7B via Ollama (LLM 1 - Fast Extraction Model):
   - Selection Rationale: Based on Mistral AI's research and benchmarks showing excellent
     performance for instruction-following and structured data extraction
   - Efficiency: 7B parameter model provides good balance between speed and quality
   - Privacy: Local deployment via Ollama ensures data privacy for sensitive CV information
   - Performance: Mistral 7B demonstrates strong instruction-following capabilities and
     better extraction quality than smaller models (e.g., Gemma 2 1B) while remaining fast
   - Use Case Fit: Ideal for fast skill extraction and requirement parsing with higher accuracy
   - Reference: Mistral AI technical reports and Hugging Face model cards
   
2. Flan-T5-XL (LLM 1 Backup):
   - Selection Rationale: Based on research from "Scaling Instruction-Finetuned Language Models"
     (Chung et al., 2022) showing Flan-T5's superior instruction-following capabilities
   - Instruction Tuning: Flan-T5-XL is specifically instruction-tuned on 1,836 tasks,
     making it excellent for structured CV generation tasks
   - Reliability: Proven track record in text generation and summarization tasks
   - Fallback Strategy: Ensures system reliability when Ollama is unavailable
   - Performance: Research shows Flan-T5-XL achieves strong results on instruction-following
     benchmarks, making it suitable for extraction tasks
   - Reference: "Scaling Instruction-Finetuned Language Models" (Chung et al., 2022)

3. GPT-4o (GPT-4 Omni) (LLM 2 - Optimization Model) - SELECTED:
   - Selection Rationale: GPT-4o is OpenAI's latest and most advanced model (as of 2024)
     with superior performance-to-cost ratio compared to GPT-4 Turbo/GPT-4.1
   - Performance: Demonstrates state-of-the-art performance in writing quality, instruction
     following, and text generation while being faster and more cost-effective
   - Cost-Effectiveness: ~50% cheaper than GPT-4 Turbo with better performance on most tasks
   - Speed: Significantly faster response times, crucial for iterative optimization loops
   - Use Case: Ideal for resume optimization requiring high-quality writing, natural skill
     incorporation, and ATS optimization
   - Quality: Produces more natural, professional, and contextually appropriate resume content
   - Availability: Widely available and stable (GPT-5 does not exist yet)
   - Reference: OpenAI GPT-4o announcement and technical documentation (May 2024)
   
   Why not GPT-4.1/Turbo?
   - More expensive than GPT-4o with similar or inferior performance
   - Slower response times
   - Older model architecture
   
   Why not GPT-5?
   - Does not exist yet (not released by OpenAI)
   - Would be overkill and prohibitively expensive if available
   - Current models (GPT-4o) are sufficient for resume optimization tasks

4. all-MiniLM-L6-v2 (Embedding Model):
   - Selection Rationale: Lightweight, fast, and effective for semantic similarity
   - Performance: 384-dimensional embeddings with good semantic understanding
   - Efficiency: Fast inference for skill matching and gap detection
   - Use Case: Semantic matching between job requirements and resume skills
   - Reference: SentenceTransformers documentation and benchmarks
   
The multi-model approach provides specialized capabilities for each pipeline stage:
- Fast, local models (LLM 1) for extraction tasks
- State-of-the-art API model (LLM 2 - GPT-4.1) for high-quality optimization
- Efficient embedding model for semantic matching
"""

import os
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

# Import refactored modules
from llm_models import OllamaModel, OpenAIModel
from prompt_builder import PromptBuilder
from content_cleaner import ContentCleaner


class CVGenerator:
    """
    Main class for generating CVs using multiple Large Language Models.
    
    Uses:
    - LLM 1 (Fast Model): Ollama (Mistral 7B) or Flan-T5-XL for extraction
    - LLM 2 (Optimization Model): GPT-4o for resume optimization
    """
    
    def __init__(self, config):
        """
        Initialize CVGenerator with configuration.
        
        Args:
            config: Config object containing model and generation parameters
        """
        self.config = config
        self.ollama_model = None
        self.flan_model = None
        self.flan_tokenizer = None
        self.openai_model = None  # LLM 2 - Optimization Model (GPT-4o)
        self.device = torch.device(config.device)
        self.primary_model = None  # Will be set after loading (LLM 1 - Fast Model)
        self.optimization_model = None  # Will be set after loading (LLM 2 - GPT-4o)
        
        # Initialize helper modules
        self.prompt_builder = PromptBuilder()
        self.content_cleaner = ContentCleaner()
        
    def load_models(self):
        """
        Load LLM models:
        - LLM 1 (Fast Model): Ollama (Mistral 7B) or Flan-T5-XL for extraction
        - LLM 2 (Optimization Model): GPT-4o for resume optimization
        """
        # Load LLM 1 (Fast Model) - for extraction tasks
        print("\n[LLM 1 - Fast Extraction Model]")
        print(f"Attempting to load Ollama ({self.config.ollama_model})...")
        try:
            self.ollama_model = OllamaModel(model_name=self.config.ollama_model)
            if self.ollama_model.available:
                print(f"✓ Ollama ({self.config.ollama_model}) loaded successfully!")
                self.primary_model = "ollama"
            else:
                print("⚠ Ollama not available, will use Flan-T5-XL")
                self.ollama_model = None
        except Exception as e:
            print(f"⚠ Could not connect to Ollama: {e}")
            self.ollama_model = None
        
        # Load Flan-T5-XL as backup for LLM 1 (only if Ollama is not available)
        if not self.ollama_model or not self.ollama_model.available:
            print(f"Loading Flan-T5-XL model: {self.config.model_name} (Ollama not available, using backup)")
            try:
                self.flan_tokenizer = T5Tokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                )
                
                self.flan_model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=getattr(torch, self.config.torch_dtype) if hasattr(torch, self.config.torch_dtype) else torch.float32
                )
                
                self.flan_model.to(self.device)
                self.flan_model.eval()
                
                print("✓ Flan-T5-XL loaded successfully!")
                self.primary_model = "flan"
                
            except Exception as e:
                print(f"Error loading Flan-T5-XL: {e}")
                raise RuntimeError("Neither Ollama nor Flan-T5-XL could be loaded for LLM 1!")
        else:
            print("✓ Ollama available - skipping Flan-T5-XL backup model")
            self.flan_model = None
            self.flan_tokenizer = None
        
        # Load LLM 2 (Optimization Model) - GPT-4o
        print("\n[LLM 2 - Optimization Model]")
        if self.config.use_openai_for_optimization:
            print(f"Loading OpenAI GPT-4o ({self.config.openai_model})...")
            print("  (Selected: Best cost-performance ratio for resume optimization)")
            try:
                api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_model = OpenAIModel(
                        model_name=self.config.openai_model,
                        api_key=api_key
                    )
                    self.optimization_model = "openai"
                    print(f"✓ OpenAI GPT-4o ({self.config.openai_model}) loaded successfully!")
                else:
                    print("⚠ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                    print("  Falling back to LLM 1 for optimization tasks.")
                    self.optimization_model = self.primary_model
            except Exception as e:
                print(f"⚠ Could not load OpenAI model: {e}")
                print("  Falling back to LLM 1 for optimization tasks.")
                self.optimization_model = self.primary_model
        else:
            print("Using LLM 1 for optimization (OpenAI disabled in config)")
            self.optimization_model = self.primary_model
    
    def generate_text(self, prompt: str, use_primary: bool = True, use_optimization_model: bool = False, **kwargs) -> str:
        """
        Generate text using the available LLM models.
        Uses primary model by default, falls back to secondary if needed.
        
        Args:
            prompt: Input prompt for text generation
            use_primary: Whether to use primary model (default: True)
            use_optimization_model: Whether to use optimization model (GPT-4o) instead (default: False)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        # Use optimization model (GPT-4o) if requested
        if use_optimization_model and self.openai_model and self.openai_model.available:
            try:
                # Filter out parameters that OpenAI doesn't need
                openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_optimization_model', 'max_new_tokens']}
                # Convert max_new_tokens to max_tokens if present
                if 'max_new_tokens' in kwargs:
                    openai_kwargs['max_tokens'] = kwargs['max_new_tokens']
                return self.openai_model.generate_text(prompt, **openai_kwargs)
            except Exception as e:
                print(f"OpenAI generation failed, falling back to primary model: {e}")
                # Fall through to primary model
        
        # Try primary model first
        if use_primary and self.primary_model == "ollama" and self.ollama_model and self.ollama_model.available:
            try:
                # Filter out parameters that Ollama doesn't need
                ollama_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_optimization_model', 'max_tokens']}
                return self.ollama_model.generate_text(prompt, **ollama_kwargs)
            except Exception as e:
                print(f"Ollama generation failed, falling back to Flan-T5-XL: {e}")
                # Fall through to Flan-T5-XL
        
        # Use Flan-T5-XL
        if self.flan_model is None or self.flan_tokenizer is None:
            raise ValueError("Flan-T5-XL model must be loaded first. Call load_models()")
        
        # Set default generation parameters
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "num_return_sequences": self.config.num_return_sequences,
        }
        
        # Filter out parameters that Flan-T5-XL doesn't accept
        # Flan-T5-XL accepts: max_new_tokens, temperature, top_p, do_sample, num_return_sequences, etc.
        # But NOT: use_optimization_model, max_tokens (use max_new_tokens instead)
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['use_optimization_model', 'max_tokens']}
        
        # Convert max_tokens to max_new_tokens if present
        if 'max_tokens' in kwargs:
            filtered_kwargs['max_new_tokens'] = kwargs['max_tokens']
        
        generation_kwargs.update(filtered_kwargs)
        
        # Tokenize input
        inputs = self.flan_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.flan_model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode output
        generated_text = self.flan_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def generate_cv_section(self, section_name: str, user_data: Dict, job_requirements: Optional[Dict] = None) -> str:
        """
        Generate a specific CV section, optionally tailored to job requirements.
        
        Args:
            section_name: Name of the CV section to generate
            user_data: Dictionary containing user information
            job_requirements: Optional job description data for tailoring
            
        Returns:
            Generated CV section text (cleaned, without section title)
        """
        # Create prompt for the section
        prompt = self._create_section_prompt(section_name, user_data, job_requirements)
        
        # Use primary model for generation
        generated_text = self.generate_text(prompt, use_primary=True)
        
        # Clean the generated content to remove section titles/headings
        cleaned_text = self._clean_section_content(generated_text, section_name)
        
        return cleaned_text
    
    def _create_section_prompt(self, section_name: str, user_data: Dict, job_requirements: Optional[Dict] = None) -> str:
        """
        Create a prompt for generating a CV section, with optional job tailoring.
        
        Args:
            section_name: Name of the section
            user_data: User information dictionary
            job_requirements: Optional job requirements for ATS optimization
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_builder.create_section_prompt(section_name, user_data, job_requirements)
    
    def optimize_resume_with_missing_skills(self, resume_text: str, user_data: Dict,
                                           missing_skills: List[str], jd_skills: List[str],
                                           job_requirements: Optional[Dict] = None) -> str:
        """
        Optimize resume using LLM 2 (GPT-4.1) by incorporating missing skills.
        
        Args:
            resume_text: Current resume text
            user_data: User information dictionary
            missing_skills: List of missing skills to incorporate
            jd_skills: List of skills from job description
            job_requirements: Optional job requirements
            
        Returns:
            Optimized resume text
        """
        prompt = f"""You are an expert ATS resume writer. Optimize this resume to include missing skills while remaining truthful.

Job Description Skills:
{', '.join(jd_skills)}

Missing Skills to Incorporate:
{', '.join(missing_skills)}

Current Resume:
{resume_text}

Rewrite the resume bullet points and sections so that:
1. They include relevant missing skills naturally and truthfully
2. Use strong action verbs (e.g., "Developed", "Implemented", "Led", "Optimized")
3. Include measurable achievements with numbers/percentages where possible
4. Remain truthful and realistic - only add skills the candidate could reasonably have
5. Optimize for ATS keyword matching
6. Maintain professional tone and formatting

Generate the optimized resume:"""
        
        # Use LLM 2 (GPT-4.1) for optimization
        optimized_resume = self.generate_text(
            prompt, 
            use_optimization_model=True, 
            max_tokens=2000,
            temperature=0.7
        )
        
        return optimized_resume
    
    def generate_tailored_cv(self, user_data: Dict, job_requirements: Optional[Dict] = None,
                            missing_skills: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate a complete, ATS-optimized CV with all sections (LLM 2 - Stronger Model).
        
        Args:
            user_data: Dictionary containing all user information
            job_requirements: Optional job description for tailoring
            missing_skills: Optional list of missing skills to incorporate
            
        Returns:
            Dictionary with section names as keys and generated text as values
        """
        cv_sections = {}
        
        print("Generating tailored CV sections using LLM 2 (Optimization Model)...")
        for section in self.config.cv_sections:
            print(f"  - Generating {section}...")
            
            # Create optimization-aware prompt if missing skills are provided
            if missing_skills and section in ["Work Experience", "Skills", "Professional Summary"]:
                prompt = self._create_optimization_prompt(section, user_data, job_requirements, missing_skills)
            else:
                prompt = self._create_section_prompt(section, user_data, job_requirements)
            
            # Use LLM 2 (GPT-4o) for optimization tasks, LLM 1 for regular generation
            if missing_skills and section in ["Work Experience", "Skills", "Professional Summary"]:
                # Reduce token limits for conciseness
                max_tokens = 600 if section == "Work Experience" else 400
                generated_content = self.generate_text(
                    prompt, 
                    use_optimization_model=True,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
            else:
                # Reduce token limits based on section type
                if section == "Work Experience":
                    max_new_tokens = 400
                elif section in ["Skills", "Education", "Certifications", "Projects", "Languages"]:
                    max_new_tokens = 200
                else:
                    max_new_tokens = 300
                generated_content = self.generate_text(prompt, use_primary=True, max_new_tokens=max_new_tokens)
            
            # Clean the generated content to remove section titles/headings
            cv_sections[section] = self.content_cleaner.clean_section_content(generated_content, section)
        
        return cv_sections
    
    
    def _create_optimization_prompt(self, section_name: str, user_data: Dict,
                                   job_requirements: Optional[Dict], missing_skills: List[str]) -> str:
        """
        Create an optimization prompt that incorporates missing skills.
        
        Args:
            section_name: Name of the section
            user_data: User information dictionary
            job_requirements: Job requirements
            missing_skills: Missing skills to incorporate
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_builder.create_optimization_prompt(
            section_name, user_data, job_requirements, missing_skills
        )
    
    def regenerate_section_with_feedback(self, section_name: str, user_data: Dict, 
                                         current_content: str, feedback: str,
                                         job_requirements: Optional[Dict] = None) -> str:
        """
        Regenerate a CV section based on user feedback.
        
        Args:
            section_name: Name of the CV section to regenerate
            user_data: Dictionary containing user information
            current_content: Current content of the section
            feedback: User feedback on what to change
            job_requirements: Optional job description for tailoring
            
        Returns:
            Regenerated CV section text
        """
        # Create feedback-aware prompt
        prompt = self._create_feedback_prompt(section_name, user_data, current_content, 
                                             feedback, job_requirements)
        
        # Use primary model for generation
        generated_text = self.generate_text(prompt, use_primary=True)
        
        # Clean the generated content to remove section titles/headings
        cleaned_text = self._clean_section_content(generated_text, section_name)
        
        return cleaned_text
    
    def _create_feedback_prompt(self, section_name: str, user_data: Dict, 
                               current_content: str, feedback: str,
                               job_requirements: Optional[Dict] = None) -> str:
        """
        Create a prompt for regenerating a CV section with user feedback.
        
        Args:
            section_name: Name of the section
            user_data: User information dictionary
            current_content: Current section content
            feedback: User feedback
            job_requirements: Optional job requirements
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Revise and improve the {section_name} section of a CV based on user feedback.\n\n"
        prompt += "IMPORTANT: Do NOT include the section title or heading in your response. Only provide the revised content.\n"
        prompt += "CRITICAL: ONLY use information provided below. Do NOT invent, add, or create any details that are not explicitly provided.\n"
        prompt += "IMPORTANT: Keep the content CONCISE and BRIEF. Aim for 2-3 pages total CV length.\n\n"
        
        # Special handling for Personal Information
        if section_name == "Personal Information":
            prompt += "CRITICAL: Output ONLY the actual contact information. Do NOT include any headings, subheadings, labels, or placeholder text.\n"
            prompt += "NEVER include the text 'Contact Information' or 'Contact Information:' anywhere in your output.\n"
            prompt += "NEVER include placeholder text like '[LinkedIn URL if provided]', '[Address if provided]', or any similar placeholders.\n"
            prompt += "The section is already titled 'Personal Information', so do NOT add any subheadings.\n"
            prompt += "Format as plain text (NO headings, NO labels, NO placeholders, just the actual information):\n"
            prompt += "Name\n"
            prompt += "Email\n"
            prompt += "Phone\n"
            prompt += "LinkedIn URL (ONLY include if actually provided in candidate data - if not provided, omit this line entirely)\n"
            prompt += "Address (ONLY include if actually provided in candidate data - if not provided, omit this line entirely)\n"
            prompt += "If LinkedIn URL or Address is not in the candidate data, do NOT mention them at all. Do NOT include placeholder text.\n\n"
        
        # Special handling for Professional Summary
        if section_name == "Professional Summary":
            prompt += "Generate EXACTLY 100 words (not more, not less).\n\n"
        
        # Special handling for Certifications
        if section_name == "Certifications":
            prompt += "List ONLY the certifications provided. Do NOT include the candidate's name.\n\n"
        
        prompt += "Candidate Information:\n"
        for key, value in user_data.items():
            if value and key != "name":  # Don't include name in certifications
                if isinstance(value, list):
                    if value:
                        prompt += f"{key}: {', '.join(str(v) for v in value)}\n"
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if v:
                            prompt += f"{key} - {k}: {v}\n"
                else:
                    prompt += f"{key}: {value}\n"
        
        prompt += "\nCurrent Section Content:\n"
        prompt += f"{current_content}\n\n"
        
        prompt += "User Feedback:\n"
        prompt += f"{feedback}\n\n"
        
        # Add user data context
        prompt += "Candidate Information:\n"
        for key, value in user_data.items():
            if value:
                if isinstance(value, list):
                    prompt += f"{key}: {', '.join(str(v) for v in value)}\n"
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if v:
                            prompt += f"{key} - {k}: {v}\n"
                else:
                    prompt += f"{key}: {value}\n"
        
        # Add job requirements for tailoring
        if job_requirements:
            prompt += "\nTarget Job Requirements:\n"
            if job_requirements.get("keywords"):
                prompt += f"Key Skills/Keywords: {', '.join(job_requirements['keywords'])}\n"
            if job_requirements.get("requirements"):
                prompt += f"Requirements: {', '.join(job_requirements['requirements'][:5])}\n"
        
        prompt += f"\nPlease regenerate ONLY the content for the {section_name} section (no title, no heading) incorporating the user feedback while maintaining professionalism and ATS-friendliness:"
        
        return prompt
    
    def regenerate_multiple_sections_with_feedback(self, cv_sections: Dict[str, str], 
                                                  user_data: Dict, feedback: Dict[str, str],
                                                  job_requirements: Optional[Dict] = None) -> Dict[str, str]:
        """
        Regenerate multiple CV sections based on user feedback.
        
        Args:
            cv_sections: Current CV sections dictionary
            user_data: Dictionary containing user information
            feedback: Dictionary mapping section names to feedback strings
            job_requirements: Optional job description for tailoring
            
        Returns:
            Updated dictionary with regenerated sections
        """
        updated_sections = cv_sections.copy()
        
        print("Regenerating sections based on feedback...")
        for section_name, section_feedback in feedback.items():
            if section_name in updated_sections:
                print(f"  - Regenerating {section_name}...")
                updated_sections[section_name] = self.regenerate_section_with_feedback(
                    section_name, user_data, updated_sections[section_name], 
                    section_feedback, job_requirements
                )
            else:
                print(f"  ⚠ Section '{section_name}' not found, skipping...")
        
        return updated_sections
    
    def format_cv_output(self, cv_sections: Dict[str, str]) -> str:
        """
        Format CV sections into a complete document.
        
        Args:
            cv_sections: Dictionary of CV sections
            
        Returns:
            Formatted CV as a string
        """
        formatted_cv = ""
        
        for section_name, section_content in cv_sections.items():
            formatted_cv += f"\n{'='*80}\n"
            formatted_cv += f"{section_name.upper()}\n"
            formatted_cv += f"{'='*80}\n"
            formatted_cv += f"{section_content}\n"
        
        return formatted_cv
