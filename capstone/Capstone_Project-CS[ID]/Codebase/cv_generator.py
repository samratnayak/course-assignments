"""
CV Generator Module

This module contains the CVGenerator class that handles CV creation using LLMs.
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()


class CVGenerator:
    """
    Main class for generating CVs using Large Language Models.
    """
    
    def __init__(self, config):
        """
        Initialize CVGenerator with configuration.
        
        Args:
            config: Config object containing model and generation parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device)
        
    def load_model(self):
        """
        Load the language model and tokenizer.
        """
        try:
            print(f"Loading model: {self.config.model_name}")
            print(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=getattr(torch, self.config.torch_dtype) if hasattr(torch, self.config.torch_dtype) else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print("Model and tokenizer loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt for text generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first. Call load_model()")
        
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
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def generate_cv_section(self, section_name: str, user_data: Dict) -> str:
        """
        Generate a specific CV section.
        
        Args:
            section_name: Name of the CV section to generate
            user_data: Dictionary containing user information
            
        Returns:
            Generated CV section text
        """
        # Create prompt for the section
        prompt = self._create_section_prompt(section_name, user_data)
        
        # Generate text
        generated_text = self.generate_text(prompt)
        
        return generated_text
    
    def _create_section_prompt(self, section_name: str, user_data: Dict) -> str:
        """
        Create a prompt for generating a CV section.
        
        Args:
            section_name: Name of the section
            user_data: User information dictionary
            
        Returns:
            Formatted prompt string
        """
        # Base prompt template
        prompt = f"Create a professional {section_name} section for a CV based on the following information:\n\n"
        
        # Add relevant user data
        for key, value in user_data.items():
            if value:
                prompt += f"{key}: {value}\n"
        
        prompt += f"\nGenerate a well-formatted {section_name} section:"
        
        return prompt
    
    def generate_full_cv(self, user_data: Dict) -> Dict[str, str]:
        """
        Generate a complete CV with all sections.
        
        Args:
            user_data: Dictionary containing all user information
            
        Returns:
            Dictionary with section names as keys and generated text as values
        """
        cv_sections = {}
        
        print("Generating CV sections...")
        for section in self.config.cv_sections:
            print(f"  - Generating {section}...")
            cv_sections[section] = self.generate_cv_section(section, user_data)
        
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
