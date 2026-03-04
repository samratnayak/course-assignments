"""
CV Generator Module

This module contains the CVGenerator class that handles CV creation using multiple LLMs.
Uses at least 2 LLMs: Gemma 3 1B via Ollama and Flan-T5-XL as backup.
"""

import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Optional
import warnings
import requests
import json

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()


class OllamaModel:
    """
    Wrapper class for Ollama API (for Gemma 3 1B model).
    """
    
    def __init__(self, model_name: str = "gemma2:1b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama model.
        
        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """
        Check if Ollama is available and running.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.available:
            raise RuntimeError("Ollama is not available. Please start Ollama service.")
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")


class CVGenerator:
    """
    Main class for generating CVs using multiple Large Language Models.
    Uses Ollama (Gemma) as primary and Flan-T5-XL as secondary model.
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
        self.device = torch.device(config.device)
        self.primary_model = None  # Will be set after loading
        
    def load_models(self):
        """
        Load both LLM models: Ollama (Gemma) and Flan-T5-XL.
        """
        # Try to load Ollama model (Gemma 3 1B)
        print("Attempting to load Ollama (Gemma 3 1B)...")
        try:
            self.ollama_model = OllamaModel(model_name="gemma2:1b")
            if self.ollama_model.available:
                print("✓ Ollama (Gemma 3 1B) loaded successfully!")
                self.primary_model = "ollama"
            else:
                print("⚠ Ollama not available, will use Flan-T5-XL only")
                self.ollama_model = None
        except Exception as e:
            print(f"⚠ Could not connect to Ollama: {e}")
            self.ollama_model = None
        
        # Load Flan-T5-XL as backup/secondary model
        print(f"Loading Flan-T5-XL model: {self.config.model_name}")
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
            
            # Set Flan as primary if Ollama is not available
            if not self.ollama_model or not self.ollama_model.available:
                self.primary_model = "flan"
            else:
                self.primary_model = "ollama"  # Use Ollama as primary
                
        except Exception as e:
            print(f"Error loading Flan-T5-XL: {e}")
            if not self.ollama_model or not self.ollama_model.available:
                raise RuntimeError("Neither Ollama nor Flan-T5-XL could be loaded!")
    
    def generate_text(self, prompt: str, use_primary: bool = True, **kwargs) -> str:
        """
        Generate text using the available LLM models.
        Uses primary model by default, falls back to secondary if needed.
        
        Args:
            prompt: Input prompt for text generation
            use_primary: Whether to use primary model (default: True)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        # Try primary model first
        if use_primary and self.primary_model == "ollama" and self.ollama_model and self.ollama_model.available:
            try:
                return self.ollama_model.generate_text(prompt, **kwargs)
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
        generation_kwargs.update(kwargs)
        
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
            Generated CV section text
        """
        # Create prompt for the section
        prompt = self._create_section_prompt(section_name, user_data, job_requirements)
        
        # Use primary model for generation
        generated_text = self.generate_text(prompt, use_primary=True)
        
        return generated_text
    
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
        # Base prompt template
        prompt = f"Create a professional, ATS-friendly {section_name} section for a CV.\n\n"
        
        # Add user data
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
        
        # Add job requirements for tailoring (ATS optimization)
        if job_requirements:
            prompt += "\nTarget Job Requirements:\n"
            if job_requirements.get("keywords"):
                prompt += f"Key Skills/Keywords: {', '.join(job_requirements['keywords'])}\n"
            if job_requirements.get("requirements"):
                prompt += f"Requirements: {', '.join(job_requirements['requirements'][:5])}\n"  # Top 5
        
            prompt += "\nTailor the section to align with these job requirements while maintaining accuracy.\n"
        
        prompt += f"\nGenerate a well-formatted, keyword-rich {section_name} section:"
        
        return prompt
    
    def generate_tailored_cv(self, user_data: Dict, job_requirements: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate a complete, ATS-optimized CV with all sections.
        
        Args:
            user_data: Dictionary containing all user information
            job_requirements: Optional job description for tailoring
            
        Returns:
            Dictionary with section names as keys and generated text as values
        """
        cv_sections = {}
        
        print("Generating tailored CV sections...")
        for section in self.config.cv_sections:
            print(f"  - Generating {section}...")
            cv_sections[section] = self.generate_cv_section(section, user_data, job_requirements)
        
        return cv_sections
    
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
