"""
Job Description Parser Module

This module parses job descriptions to extract requirements, responsibilities, and keywords.
"""

import re
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


class JobDescriptionParser:
    """
    Class to parse job descriptions and extract key information.
    """
    
    def __init__(self, llm_model=None):
        """
        Initialize JobDescriptionParser.
        
        Args:
            llm_model: Optional LLM model for advanced parsing
        """
        self.llm_model = llm_model
    
    def parse_job_description(self, job_text: str) -> Dict:
        """
        Parse job description to extract requirements and keywords.
        
        Args:
            job_text: Job description text
            
        Returns:
            Dictionary with parsed job information
        """
        parsed_data = {
            "job_title": "",
            "company": "",
            "requirements": [],
            "responsibilities": [],
            "keywords": [],
            "skills_required": [],
            "experience_required": ""
        }
        
        # Extract job title (usually in first few lines)
        lines = job_text.split('\n')[:5]
        for line in lines:
            if 'title' in line.lower() or 'position' in line.lower():
                parsed_data["job_title"] = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                break
        
        # Extract keywords (common technical terms)
        keywords = self._extract_keywords(job_text)
        parsed_data["keywords"] = keywords
        parsed_data["skills_required"] = keywords  # Skills are often keywords
        
        # Extract requirements
        requirements = self._extract_section(job_text, ["requirement", "qualification", "must have"])
        parsed_data["requirements"] = requirements
        
        # Extract responsibilities
        responsibilities = self._extract_section(job_text, ["responsibilit", "duties", "role"])
        parsed_data["responsibilities"] = responsibilities
        
        # Extract experience requirement
        experience = self._extract_experience_requirement(job_text)
        parsed_data["experience_required"] = experience
        
        # Use LLM for better extraction if available
        if self.llm_model:
            parsed_data = self._llm_enhanced_parsing(job_text, parsed_data)
        
        return parsed_data
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from job description.
        
        Args:
            text: Job description text
            
        Returns:
            List of keywords
        """
        # Common technical keywords
        common_keywords = [
            "python", "java", "javascript", "sql", "machine learning", "ai", "deep learning",
            "data science", "cloud", "aws", "azure", "docker", "kubernetes", "git",
            "agile", "scrum", "api", "rest", "microservices", "react", "node.js",
            "tensorflow", "pytorch", "nlp", "computer vision", "big data", "hadoop"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in common_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_section(self, text: str, section_keywords: List[str]) -> List[str]:
        """
        Extract a specific section from job description.
        
        Args:
            text: Job description text
            section_keywords: Keywords to identify the section
            
        Returns:
            List of items in the section
        """
        items = []
        lines = text.split('\n')
        in_section = False
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in section_keywords):
                in_section = True
                continue
            
            if in_section:
                if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•') or 
                                   line.strip()[0].isdigit()):
                    items.append(line.strip().lstrip('- •0123456789. '))
                elif line.strip() and not any(other in line_lower for other in 
                    ["requirement", "responsibilit", "qualification", "skill", "benefit"]):
                    # End of section
                    break
        
        return items
    
    def _extract_experience_requirement(self, text: str) -> str:
        """
        Extract experience requirement from job description.
        
        Args:
            text: Job description text
            
        Returns:
            Experience requirement string
        """
        # Look for patterns like "X years", "X+ years", etc.
        pattern = r'(\d+[\+\-]?\s*years?[\s\w]*experience)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
        return ""
    
    def _llm_enhanced_parsing(self, text: str, initial_data: Dict) -> Dict:
        """
        Use LLM to enhance job description parsing.
        
        Args:
            text: Job description text
            initial_data: Initially parsed data
            
        Returns:
            Enhanced parsed data
        """
        if not self.llm_model:
            return initial_data
        
        prompt = f"""Extract key information from this job description:

{text}

Extract:
1. Job Title
2. Required Skills (as a list)
3. Key Requirements
4. Main Responsibilities
5. Experience Required

Format as structured text."""
        
        try:
            enhanced_text = self.llm_model.generate_text(prompt)
            # Parse enhanced text and update initial_data
            # This is a simplified version - can be enhanced
        except:
            pass  # Fallback to initial parsing
        
        return initial_data
