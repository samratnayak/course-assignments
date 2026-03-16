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
    Can detect and extract job descriptions from input files or standalone text.
    """
    
    def __init__(self, llm_model=None):
        """
        Initialize JobDescriptionParser.
        
        Args:
            llm_model: Optional LLM model for advanced parsing
        """
        self.llm_model = llm_model
    
    def detect_job_description_in_text(self, text: str) -> Optional[str]:
        """
        Detect if a job description is present in the input text.
        Looks for common job description markers and sections.
        
        Args:
            text: Input text that may contain both profile and job description
            
        Returns:
            Extracted job description text if found, None otherwise
        """
        # Common job description section markers
        job_markers = [
            "JOB DESCRIPTION",
            "JOB POSTING",
            "POSITION DESCRIPTION",
            "CAREER OPPORTUNITY",
            "WE ARE HIRING",
            "JOB REQUIREMENTS",
            "POSITION REQUIREMENTS",
            "ROLE DESCRIPTION",
            "VACANCY",
            "OPENING"
        ]
        
        text_upper = text.upper()
        job_description_start = None
        
        # Find the start of job description section
        for marker in job_markers:
            marker_pos = text_upper.find(marker)
            if marker_pos != -1:
                # Look for the actual marker in original case
                for i in range(max(0, marker_pos - 50), min(len(text), marker_pos + 200)):
                    if text[i:i+len(marker)].upper() == marker:
                        job_description_start = i
                        break
                if job_description_start is not None:
                    break
        
        if job_description_start is not None:
            # Extract job description (from marker to end, or until next major section)
            # Look for common profile section markers that might come after
            profile_markers = [
                "\n\nEDUCATION",
                "\n\nWORK EXPERIENCE",
                "\n\nSKILLS",
                "\n\nPROJECTS",
                "\n\nCERTIFICATIONS"
            ]
            
            job_text = text[job_description_start:]
            
            # Try to find where profile starts (if job description is first)
            profile_start = len(job_text)
            for marker in profile_markers:
                marker_pos = job_text.upper().find(marker.upper())
                if marker_pos != -1 and marker_pos < profile_start:
                    profile_start = marker_pos
            
            # Extract job description (everything before profile sections)
            job_description = job_text[:profile_start].strip()
            
            # Validate: job description should be substantial (at least 100 chars)
            if len(job_description) >= 100:
                return job_description
        
        return None
    
    def separate_profile_and_job_description(self, text: str) -> tuple:
        """
        Separate profile data and job description from combined input text.
        
        Args:
            text: Combined text containing both profile and job description
            
        Returns:
            Tuple of (profile_text, job_description_text or None)
        """
        job_desc = self.detect_job_description_in_text(text)
        
        if job_desc:
            # Remove job description from profile text
            profile_text = text.replace(job_desc, "").strip()
            return profile_text, job_desc
        
        return text, None
    
    def parse_job_description(self, job_text: str, from_input_file: bool = False) -> Dict:
        """
        Parse job description to extract requirements and keywords.
        
        Args:
            job_text: Job description text
            from_input_file: Whether the job description was extracted from input file
            
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
        Use LLM 1 (Fast Model) to extract skills and requirements from job description.
        
        Args:
            text: Job description text
            initial_data: Initially parsed data
            
        Returns:
            Enhanced parsed data with structured skills
        """
        if not self.llm_model:
            return initial_data
        
        prompt = f"""Extract skills and requirements from this job description. Return JSON-like structure:

Job Description:
{text}

Extract and return in this exact format:
{{
  "skills": ["Python", "Machine Learning", "Docker"],
  "tools": ["TensorFlow", "Git", "AWS"],
  "soft_skills": ["communication", "teamwork"],
  "job_title": "Job Title",
  "requirements": ["Requirement 1", "Requirement 2"],
  "responsibilities": ["Responsibility 1", "Responsibility 2"]
}}

Focus on extracting ALL required skills, tools, and technologies. Be comprehensive."""
        
        try:
            enhanced_text = self.llm_model.generate_text(prompt, max_new_tokens=512)
            # Try to parse JSON from LLM output
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', enhanced_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    # Update initial_data with LLM-extracted skills
                    if "skills" in parsed:
                        initial_data["keywords"] = parsed["skills"] + parsed.get("tools", [])
                        initial_data["skills_required"] = parsed["skills"] + parsed.get("tools", [])
                    if "job_title" in parsed:
                        initial_data["job_title"] = parsed["job_title"]
                    if "requirements" in parsed:
                        initial_data["requirements"] = parsed["requirements"]
                    if "responsibilities" in parsed:
                        initial_data["responsibilities"] = parsed["responsibilities"]
                except json.JSONDecodeError:
                    pass  # Fallback to initial parsing
        except Exception as e:
            print(f"Warning: LLM parsing failed: {e}")
            pass  # Fallback to initial parsing
        
        return initial_data
