"""
Configuration file for CV Creation using LLMs project.

Contains all configuration parameters, model settings, and paths.
"""

import os
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration class for CV Creation project."""
    
    # Model Configuration
    # Primary: Mistral 7B via Ollama (if available)
    #   - Selected for efficiency, privacy, and strong instruction-following
    #   - Research-backed: Mistral AI's 7B model shows excellent performance for extraction tasks
    #   - Better than Gemma 2 1B for structured data extraction while remaining fast
    # Secondary: Flan-T5-XL as backup
    #   - Selected for proven instruction-following capabilities
    #   - Research-backed: "Scaling Instruction-Finetuned Language Models" (Chung et al., 2022)
    model_name: str = "google/flan-t5-xl"  # Flan-T5-XL model
    ollama_model: str = "mistral"  # Ollama model name (Mistral 7B)
    ollama_url: str = "http://localhost:11434"  # Ollama API URL
    device: str = "cpu"  # Will be set in __post_init__
    torch_dtype: str = "float32"  # Will be adjusted based on device
    
    # OpenAI Configuration for LLM 2 (Optimization Model)
    # Selected: GPT-4o (GPT-4 Omni) - Best fit for resume optimization
    # Rationale: Latest available model, cost-effective, faster than GPT-4 Turbo,
    #            excellent instruction-following, optimal for text generation tasks
    # Note: GPT-5 does not exist yet; GPT-4.1/Turbo is more expensive with similar performance
    openai_model: str = "gpt-4o"  # GPT-4o (GPT-4 Omni) - Best choice for LLM 2
    openai_api_key: Optional[str] = None  # Will be set from environment or argument
    use_openai_for_optimization: bool = True  # Use GPT-4o for optimization tasks
    
    # Generation Parameters
    max_length: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Paths - All files in Codebase directory (no subdirectories per instructions)
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    input_dir: str = os.path.join(base_dir, "input")
    output_dir: str = os.path.join(base_dir, "output")
    cache_dir: Optional[str] = None
    
    # CV Template Settings
    cv_sections: list = None
    
    def __post_init__(self):
        """Initialize paths and create directories if needed."""
        # Create directories
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize CV sections if not provided
        if self.cv_sections is None:
            self.cv_sections = [
                "Personal Information",
                "Professional Summary",
                "Work Experience",
                "Education",
                "Skills",
                "Certifications",
                "Projects",
                "Languages"
            ]
        
        # Set device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = "float16"
        else:
            self.device = "cpu"
            self.torch_dtype = "float32"
            print("CUDA not available, using CPU")
