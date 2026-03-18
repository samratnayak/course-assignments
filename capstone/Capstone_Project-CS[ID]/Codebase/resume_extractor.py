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
  "name": "Full Name of the Candidate (first line of resume, not project names)",
  "contact": {{
    "email": "email@example.com",
    "phone": "1234567890"
  }},
  "education": ["Degree, Institution, Year"],
  "experience": ["Job Title, Company, Duration"],
  "projects": ["Project Name - Description"]
}}

IMPORTANT: 
- Extract the candidate's FULL NAME from the first line or header of the resume
- Do NOT confuse project names or section headers with the candidate name
- The name is typically the first line before contact information
- For phone numbers: Use the EXACT format from the input. Do NOT add country codes, ISD codes (like +1, +91, +44), or '+' prefix. 
  If the input shows '8866835339', extract it as '8866835339' without adding any country code.
- Focus on extracting ALL skills, tools, and technologies mentioned. Be comprehensive."""
        
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
        
        # Fallback to text parsing if JSON parsing fails or name is invalid
        json_name = structured_data.get("name", "")
        json_name_valid = json_name and self._is_valid_name(json_name)
        
        if not structured_data.get("skills") or not json_name_valid:
            fallback_data = self._parse_extracted_text(extracted_text, text)
            # Merge: use JSON skills if available, but prefer fallback name if JSON name is invalid
            if structured_data.get("skills"):
                fallback_data["skills"] = structured_data["skills"]
                fallback_data["tools"] = structured_data.get("tools", [])
                fallback_data["soft_skills"] = structured_data.get("soft_skills", [])
            
            # Use fallback name if JSON name is missing or invalid
            fallback_name = fallback_data.get("name", "")
            fallback_name_valid = fallback_name and self._is_valid_name(fallback_name)
            
            if not json_name_valid:
                if fallback_name_valid:
                    # Use fallback name and merge other fields
                    structured_data["name"] = fallback_name
                    structured_data.update({k: v for k, v in fallback_data.items() if k != "name"})
                else:
                    # Both names invalid - clear the name, will use "Candidate" as default
                    structured_data["name"] = ""
            else:
                # JSON name is valid, but merge other fields from fallback
                structured_data.update({k: v for k, v in fallback_data.items() if k != "name"})
        
        # Final validation: ensure name is valid, otherwise clear it
        final_name = structured_data.get("name", "")
        if final_name and not self._is_valid_name(final_name):
            structured_data["name"] = ""
        
        return structured_data
    
    def _is_valid_name(self, name: str) -> bool:
        """
        Check if extracted name is likely valid (not a project name, section header, etc.).
        
        Args:
            name: Extracted name string
            
        Returns:
            True if name appears valid, False otherwise
        """
        if not name or len(name) > 50:
            return False
        
        name_lower = name.lower().strip()
        name_lower = re.sub(r'[^\w\s]', '', name_lower)  # Remove punctuation for comparison
        
        # Check for common project/section keywords (more comprehensive)
        invalid_keywords = ['classifier', 'platform', 'system', 'application', 'project', 
                          'model', 'education', 'experience', 'skills', 'certifications',
                          'projects', 'achievements', 'languages', 'machine learning',
                          'machine', 'learning', 'deep learning', 'neural network',
                          'e-commerce', 'sentiment analysis', 'analysis', 'algorithm',
                          'software', 'development', 'framework', 'library', 'api',
                          'database', 'server', 'client', 'web', 'mobile', 'app']
        if any(keyword in name_lower for keyword in invalid_keywords):
            return False
        
        # Check if it's all caps and long (likely a section header)
        if name.isupper() and len(name) > 5:
            return False
        
        # Check if it has too many words (likely not a name)
        words = name.split()
        if len(words) > 4:
            return False
        
        # Check if it looks like a technical term (contains common tech words)
        tech_words = ['python', 'java', 'javascript', 'react', 'angular', 'node', 'sql',
                     'html', 'css', 'docker', 'kubernetes', 'aws', 'azure', 'gcp']
        if any(tech_word in name_lower for tech_word in tech_words):
            return False
        
        # Check if it's too short (less than 2 characters) or too long for a name
        if len(name.strip()) < 2:
            return False
        
        # Check if it contains mostly numbers or special characters
        if sum(c.isalnum() for c in name) < len(name) * 0.5:
            return False
        
        return True
    
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
                # Clean and validate name extraction
                extracted_name = parsed.get("name", "").strip().strip('"\'')
                # Remove trailing commas and quotes
                extracted_name = extracted_name.rstrip(',').strip().strip('"\'')
                # Validate it's not a project name or section header
                if extracted_name and self._is_valid_name(extracted_name):
                    structured_data["name"] = extracted_name
                
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
        if not structured_data["name"] or not self._is_valid_name(structured_data["name"]):
            if original_text:
                # Try to extract name from first few lines (before sections start)
                lines = original_text.split('\n')
                # Look for name in first 5 lines (before section headers)
                for i, line in enumerate(lines[:5]):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip if it looks like a section header (all caps, contains keywords)
                    line_upper = line.upper()
                    if any(keyword in line_upper for keyword in ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 
                                                                 'CERTIFICATIONS', 'OBJECTIVE', 'SUMMARY', 'PROFILE']):
                        break
                    
                    # Skip if it contains email pattern
                    if '@' in line or 'email' in line.lower():
                        continue
                    
                    # Skip if it contains phone pattern
                    if any(char in line for char in ['+', '-', '(', ')']) and any(char.isdigit() for char in line):
                        continue
                    
                    # Skip if it's a URL
                    if 'http' in line.lower() or 'www.' in line.lower():
                        continue
                    
                    # Skip if it contains common resume section indicators
                    if any(indicator in line.lower() for indicator in ['resume', 'cv', 'curriculum vitae']):
                        continue
                    
                    # If line looks like a name (1-4 words, reasonable length)
                    words = line.split()
                    if 1 <= len(words) <= 4 and len(line) < 50:
                        # Check if it's likely a name (not a project title or section)
                        # Names typically don't have all caps words (except initials)
                        has_all_caps = any(word.isupper() and len(word) > 2 for word in words)
                        if not has_all_caps:
                            # Validate the extracted name
                            if self._is_valid_name(line):
                                structured_data["name"] = line
                                break
        
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
