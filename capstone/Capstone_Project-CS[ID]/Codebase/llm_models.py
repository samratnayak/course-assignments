"""
LLM Model Wrappers

This module contains wrapper classes for different LLM providers:
- OllamaModel: For local Ollama models (Mistral 7B, etc.)
- OpenAIModel: For OpenAI API models (GPT-4o, etc.)
"""

import os
import requests
from typing import Optional

# Try to import OpenAI (optional dependency)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OllamaModel:
    """
    Wrapper class for Ollama API (for local models like Mistral 7B).
    
    Attributes:
        model_name: Name of the Ollama model
        base_url: Base URL for Ollama API
        available: Whether Ollama is available and running
    """
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
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
        except Exception:
            return False
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If Ollama is not available or API call fails
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


class OpenAIModel:
    """
    Wrapper class for OpenAI API (for GPT-4o and other OpenAI models).
    
    Attributes:
        model_name: Name of the OpenAI model
        api_key: OpenAI API key
        client: OpenAI client instance
        available: Whether the model is available
    """
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model (default: gpt-4o)
            api_key: OpenAI API key (if None, will try to get from environment)
            
        Raises:
            RuntimeError: If OpenAI package is not installed
            ValueError: If API key is not provided
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.available = True
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If model is not available or API call fails
        """
        if not self.available:
            raise RuntimeError("OpenAI model is not available.")
        
        # Default parameters for GPT-4
        default_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2000)),
        }
        default_params.update({k: v for k, v in kwargs.items() if k not in ["max_new_tokens"]})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert ATS resume writer and career advisor."
                    },
                    {"role": "user", "content": prompt}
                ],
                **default_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")
