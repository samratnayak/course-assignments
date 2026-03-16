"""
Resume Data Extractor Module

This module uses LLMs to extract structured information from unstructured resume data.
Extracts: personal details, education, work history, skills, achievements, and projects.
"""

import json
import re
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings("ignore")


class ResumeExtractor:
    """
    Class to extract structured resume information using LLMs.
    """
    
    def __init__(self, llm_model):
        """
        Initialize ResumeExtractor with an LLM model.
        
        Args:
            llm_model: LLM model instance (from cv_generator or ollama)
        """
        self.llm_model = llm_model
    
    def create_extraction_prompt(self, text: str) -> str:
        """
        Create a prompt for extracting resume information (LLM 1 - Fast Model).
        Focuses on extracting skills and requirements in structured JSON format.
        
        Args:
            text: Unstructured resume text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Extract skills and requirements from this resume text. Return a JSON-like structure:

Resume Text:
{text}

Extract and return in this exact format:
{{
  "skills": ["Python", "Machine Learning", "Docker"],
  "tools": ["TensorFlow", "Git", "AWS"],
  "soft_skills": ["communication", "teamwork", "leadership"],
  "name": "Full Name",
  "contact": {{
    "email": "email@example.com",
    "phone": "+1234567890"
  }},
  "education": ["Degree, Institution, Year"],
  "experience": ["Job Title, Company, Duration"],
  "projects": ["Project Name - Description"]
}}

Focus on extracting ALL skills, tools, and technologies mentioned. Be comprehensive."""
        
        return prompt
    
    def extract_resume_data(self, text: str) -> Dict:
        """
        Extract structured resume data from unstructured text using LLM 1 (Fast Model).
        
        Args:
            text: Unstructured resume text
            
        Returns:
            Dictionary containing extracted resume information with structured skills
        """
        # Create extraction prompt
        prompt = self.create_extraction_prompt(text)
        
        # Use LLM 1 (fast model) to extract information
        try:
            extracted_text = self.llm_model.generate_text(prompt, max_new_tokens=512)
        except Exception as e:
            print(f"Warning: LLM extraction failed, using fallback parsing: {e}")
            extracted_text = ""
        
        # Try to parse JSON from LLM output
        structured_data = self._parse_json_extraction(extracted_text, text)
        
        # Fallback to text parsing if JSON parsing fails
        if not structured_data.get("skills"):
            structured_data = self._parse_extracted_text(extracted_text, text)
        
        return structured_data
    
    def _parse_json_extraction(self, extracted_text: str, original_text: str) -> Dict:
        """
        Parse JSON-like output from LLM extraction.
        
        Args:
            extracted_text: LLM output text
            original_text: Original resume text for fallback
            
        Returns:
            Structured dictionary
        """
        structured_data = {
            "name": "",
            "contact": {"email": "", "phone": "", "address": ""},
            "education": [],
            "experience": [],
            "skills": [],
            "tools": [],
            "soft_skills": [],
            "achievements": [],
            "projects": [],
            "certifications": [],
            "languages": []
        }
        
        # Try to extract JSON from the text
        json_match = re.search(r'\{[^{}]*\}', extracted_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Map parsed data to structured format
                structured_data["skills"] = parsed.get("skills", [])
                structured_data["tools"] = parsed.get("tools", [])
                structured_data["soft_skills"] = parsed.get("soft_skills", [])
                structured_data["name"] = parsed.get("name", "")
                
                if "contact" in parsed:
                    structured_data["contact"].update(parsed["contact"])
                
                structured_data["education"] = parsed.get("education", [])
                structured_data["experience"] = parsed.get("experience", [])
                structured_data["projects"] = parsed.get("projects", [])
                
            except json.JSONDecodeError:
                pass  # Fall back to text parsing
        
        # Combine all skills into a single list for easier processing
        all_skills = structured_data["skills"] + structured_data["tools"]
        structured_data["all_skills"] = all_skills
        
        return structured_data
    
    def _parse_extracted_text(self, extracted_text: str, original_text: str) -> Dict:
        """
        Parse LLM-extracted text into structured dictionary.
        
        Args:
            extracted_text: Text generated by LLM
            original_text: Original resume text for fallback
            
        Returns:
            Structured dictionary with resume fields
        """
        # Initialize structured data dictionary
        structured_data = {
            "name": "",
            "contact": {
                "email": "",
                "phone": "",
                "address": ""
            },
            "education": [],
            "experience": [],
            "skills": [],
            "achievements": [],
            "projects": [],
            "certifications": [],
            "languages": []
        }
        
        # Simple parsing logic - can be enhanced with more sophisticated parsing
        lines = extracted_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            line_lower = line.lower()
            if 'name' in line_lower and ':' in line:
                structured_data["name"] = line.split(':', 1)[1].strip()
            elif 'email' in line_lower:
                structured_data["contact"]["email"] = line.split(':', 1)[1].strip() if ':' in line else ""
            elif 'phone' in line_lower:
                structured_data["contact"]["phone"] = line.split(':', 1)[1].strip() if ':' in line else ""
            elif 'education' in line_lower:
                current_section = "education"
            elif 'experience' in line_lower or 'work' in line_lower:
                current_section = "experience"
            elif 'skill' in line_lower:
                current_section = "skills"
            elif 'achievement' in line_lower or 'award' in line_lower:
                current_section = "achievements"
            elif 'project' in line_lower:
                current_section = "projects"
            elif 'certification' in line_lower:
                current_section = "certifications"
            elif 'language' in line_lower:
                current_section = "languages"
            elif current_section and line.startswith('-') or line.startswith('•'):
                # Add item to current section
                item = line.lstrip('- •').strip()
                if item and current_section in structured_data:
                    if isinstance(structured_data[current_section], list):
                        structured_data[current_section].append(item)
        
        # If extraction failed, use original text as fallback
        if not structured_data["name"] and original_text:
            # Try to extract name from first line
            first_line = original_text.split('\n')[0].strip()
            if len(first_line) < 100:  # Likely a name
                structured_data["name"] = first_line
        
        return structured_data
    
    def extract_to_json(self, text: str, output_path: Optional[str] = None) -> Dict:
        """
        Extract resume data and optionally save to JSON file.
        
        Args:
            text: Unstructured resume text
            output_path: Optional path to save JSON file
            
        Returns:
            Dictionary with extracted resume data
        """
        structured_data = self.extract_resume_data(text)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        return structured_data
