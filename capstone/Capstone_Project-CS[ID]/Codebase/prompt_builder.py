"""
Prompt Builder Module

This module handles creation of prompts for CV generation, optimization, and feedback.
All prompt generation logic is centralized here for maintainability.
"""

from typing import Dict, List, Optional


class PromptBuilder:
    """
    Builder class for creating prompts for CV generation.
    
    Methods:
        create_section_prompt: Create prompt for generating a CV section
        create_optimization_prompt: Create prompt for optimizing a section with missing skills
        create_feedback_prompt: Create prompt for regenerating a section with user feedback
    """
    
    # Base prompt templates
    BASE_SECTION_PROMPT = (
        "Create a professional, ATS-friendly {section_name} section for a CV.\n\n"
        "IMPORTANT: Do NOT include the section title or heading in your response. "
        "Only provide the content.\n"
        "CRITICAL: ONLY use information provided below. Do NOT invent, add, or create "
        "any details that are not explicitly provided.\n"
        "IMPORTANT: Keep the content CONCISE and BRIEF. Aim for 2-3 pages total CV length. "
        "Be selective and include only the most relevant information.\n\n"
    )
    
    BASE_OPTIMIZATION_PROMPT = (
        "Create an ATS-optimized {section_name} section for a CV.\n\n"
        "IMPORTANT: Do NOT include the section title or heading in your response. "
        "Only provide the content.\n"
        "CRITICAL: ONLY use information provided below. Do NOT invent, add, or create "
        "any details that are not explicitly provided.\n"
        "IMPORTANT: Keep the content CONCISE and BRIEF. Aim for 2-3 pages total CV length. "
        "Be selective and include only the most relevant information.\n\n"
    )
    
    BASE_FEEDBACK_PROMPT = (
        "Revise and improve the {section_name} section of a CV based on user feedback.\n\n"
        "IMPORTANT: Do NOT include the section title or heading in your response. "
        "Only provide the revised content.\n"
        "CRITICAL: ONLY use information provided below. Do NOT invent, add, or create "
        "any details that are not explicitly provided.\n"
        "IMPORTANT: Keep the content CONCISE and BRIEF. Aim for 2-3 pages total CV length.\n\n"
    )
    
    @staticmethod
    def create_section_prompt(
        section_name: str,
        user_data: Dict,
        job_requirements: Optional[Dict] = None
    ) -> str:
        """
        Create a prompt for generating a CV section.
        
        Args:
            section_name: Name of the section
            user_data: User information dictionary
            job_requirements: Optional job requirements for ATS optimization
            
        Returns:
            Formatted prompt string
        """
        prompt = PromptBuilder.BASE_SECTION_PROMPT.format(section_name=section_name)
        
        # Add section-specific instructions
        prompt += PromptBuilder._get_section_specific_instructions(section_name)
        
        # Add candidate information
        prompt += PromptBuilder._format_candidate_information(user_data, exclude_name=True)
        
        # Add job requirements if provided
        if job_requirements:
            prompt += PromptBuilder._format_job_requirements(job_requirements)
        
        prompt += f"\nGenerate ONLY the content for the {section_name} section "
        prompt += "(no title, no heading, just the content):"
        
        return prompt
    
    @staticmethod
    def create_optimization_prompt(
        section_name: str,
        user_data: Dict,
        job_requirements: Optional[Dict],
        missing_skills: List[str]
    ) -> str:
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
        prompt = PromptBuilder.BASE_OPTIMIZATION_PROMPT.format(section_name=section_name)
        
        # Add section-specific instructions
        prompt += PromptBuilder._get_section_specific_instructions(section_name, is_optimization=True)
        
        # Add candidate information
        prompt += PromptBuilder._format_candidate_information(
            user_data, exclude_name=True, exclude_all_skills=True
        )
        
        # Add job requirements if provided
        if job_requirements:
            prompt += PromptBuilder._format_job_requirements(job_requirements, include_keywords_only=True)
        
        # Add missing skills with emphasis
        prompt += f"\nCRITICAL - Missing Skills to Incorporate: {', '.join(missing_skills)}\n"
        prompt += "These skills are REQUIRED for the job. You MUST find ways to incorporate them "
        prompt += "truthfully based on the candidate's background. Look for:\n"
        prompt += "- Related skills or experiences that demonstrate these capabilities\n"
        prompt += "- Transferable skills from similar technologies or domains\n"
        prompt += "- Projects or work that used similar concepts\n\n"
        
        # Add optimization instructions with emphasis on ATS
        prompt += f"\nGenerate ONLY the content for the {section_name} section "
        prompt += "(no title, no heading) that:\n"
        prompt += "- AGGRESSIVELY incorporates ALL relevant missing skills "
        prompt += "(find truthful connections to candidate's experience)\n"
        prompt += "- Uses STRONG action verbs and ATS-friendly keywords from job description\n"
        prompt += "- Includes quantifiable achievements and metrics (from provided information)\n"
        prompt += "- MAXIMIZES keyword density for ATS matching (use job description keywords naturally)\n"
        prompt += "- Prioritizes job-relevant information over generic content\n"
        prompt += "- Does NOT invent new job titles, companies, or experiences\n"
        prompt += "- Focus on improving ATS score by matching job requirements closely\n"
        prompt += f"\n{section_name} content (no title):"
        
        return prompt
    
    @staticmethod
    def create_feedback_prompt(
        section_name: str,
        user_data: Dict,
        current_content: str,
        feedback: str,
        job_requirements: Optional[Dict] = None
    ) -> str:
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
        prompt = PromptBuilder.BASE_FEEDBACK_PROMPT.format(section_name=section_name)
        
        # Add section-specific instructions
        prompt += PromptBuilder._get_section_specific_instructions(section_name)
        
        # Add current content
        prompt += "Current Section Content:\n"
        prompt += f"{current_content}\n\n"
        
        # Add user feedback
        prompt += "User Feedback:\n"
        prompt += f"{feedback}\n\n"
        
        # Add candidate information
        prompt += PromptBuilder._format_candidate_information(user_data, exclude_name=True)
        
        # Add job requirements if provided
        if job_requirements:
            prompt += PromptBuilder._format_job_requirements(job_requirements)
        
        prompt += "\nRevise the section based on the feedback above. "
        prompt += "Maintain accuracy and only use information from Candidate Information."
        
        return prompt
    
    @staticmethod
    def _get_section_specific_instructions(
        section_name: str,
        is_optimization: bool = False
    ) -> str:
        """
        Get section-specific instructions for prompts.
        
        Args:
            section_name: Name of the section
            is_optimization: Whether this is for optimization (affects some instructions)
            
        Returns:
            Section-specific instruction string
        """
        instructions = ""
        
        if section_name == "Personal Information":
            instructions = PromptBuilder._get_personal_information_instructions()
        elif section_name == "Work Experience":
            instructions = PromptBuilder._get_work_experience_instructions()
        elif section_name == "Professional Summary":
            instructions = PromptBuilder._get_professional_summary_instructions(is_optimization)
        elif section_name == "Skills":
            instructions = PromptBuilder._get_skills_instructions()
        elif section_name == "Education":
            instructions = PromptBuilder._get_education_instructions()
        elif section_name == "Projects":
            instructions = PromptBuilder._get_projects_instructions()
        elif section_name == "Certifications":
            instructions = PromptBuilder._get_certifications_instructions()
        elif section_name == "Languages":
            instructions = PromptBuilder._get_languages_instructions()
        
        return instructions
    
    @staticmethod
    def _get_personal_information_instructions() -> str:
        """Get instructions for Personal Information section."""
        return (
            "CRITICAL: Output ONLY the actual contact information. "
            "Do NOT include any headings, subheadings, labels, or placeholder text.\n"
            "NEVER include the text 'Contact Information' or 'Contact Information:' "
            "anywhere in your output.\n"
            "NEVER include placeholder text like '[LinkedIn URL if provided]', "
            "'[Address if provided]', or any similar placeholders.\n"
            "ABSOLUTELY FORBIDDEN: Do NOT include any explanations, reasoning, notes, or commentary. "
            "Do NOT include sentences starting with 'Please note', 'However', 'Therefore', 'While', "
            "'It should be noted', 'Note that', or any similar explanatory phrases.\n"
            "Do NOT include any analysis, observations, or meta-commentary about the candidate's information.\n"
            "Do NOT include any sentences about experience, qualifications, skills, or job requirements.\n"
            "The section is already titled 'Personal Information', so do NOT add any subheadings.\n"
            "Format as plain text (NO headings, NO labels, NO placeholders, NO explanations, just the actual information):\n"
            "Name\n"
            "Email\n"
            "Phone (CRITICAL: Use the EXACT phone number format from the input. Do NOT add country codes, "
            "ISD codes, or formatting like +1, +91, etc. If the input shows '8867830338', output it as '8867830338' "
            "or with the same formatting as provided. Do NOT add '+' prefix or country codes.)\n"
            "LinkedIn URL (ONLY include if actually provided in candidate data - "
            "if not provided, omit this line entirely)\n"
            "Address (ONLY include if actually provided in candidate data - "
            "if not provided, omit this line entirely)\n"
            "If LinkedIn URL or Address is not in the candidate data, do NOT mention them at all. "
            "Do NOT include placeholder text.\n"
            "REMEMBER: Only output contact information. No explanations, no reasoning, no notes, no commentary, no analysis.\n\n"
        )
    
    @staticmethod
    def _get_work_experience_instructions() -> str:
        """Get instructions for Work Experience section."""
        return (
            "CRITICAL: List ONLY the 3-4 most recent and relevant positions. For each position:\n"
            "- Include: Position Title, Company Name, Dates (MM/YYYY - MM/YYYY or Present)\n"
            "- Include 2-3 concise bullet points per position (maximum 1 line each)\n"
            "- Focus on achievements and impact, not just responsibilities\n"
            "- Use strong action verbs and quantifiable results\n"
            "- Keep each bullet point under 80 characters when possible\n\n"
        )
    
    @staticmethod
    def _get_professional_summary_instructions(is_optimization: bool = False) -> str:
        """Get instructions for Professional Summary section."""
        if is_optimization:
            return "Generate EXACTLY 100 words (not more, not less).\n\n"
        return (
            "Generate EXACTLY 100 words (not more, not less). "
            "Summarize the candidate's experience, skills, and career objectives "
            "based ONLY on the provided information.\n\n"
        )
    
    @staticmethod
    def _get_skills_instructions() -> str:
        """Get instructions for Skills section."""
        return (
            "List skills CONCISELY. Group related skills together. Maximum 2-3 lines total.\n"
            "Format as: Category: Skill1, Skill2, Skill3 | Category2: Skill4, Skill5\n\n"
        )
    
    @staticmethod
    def _get_education_instructions() -> str:
        """Get instructions for Education section."""
        return (
            "List education CONCISELY. Include: Degree, Institution, Year, GPA (if notable). "
            "Maximum 2-3 entries.\n\n"
        )
    
    @staticmethod
    def _get_projects_instructions() -> str:
        """Get instructions for Projects section."""
        return (
            "List ONLY the 2-3 most relevant projects. For each project:\n"
            "- Include: Project Name, Brief description (1 line), Key technologies used\n"
            "- Keep each project to 2-3 lines maximum\n\n"
        )
    
    @staticmethod
    def _get_certifications_instructions() -> str:
        """Get instructions for Certifications section."""
        return (
            "List ONLY the certifications provided. Do NOT include the candidate's name. "
            "Format as: Certification Name (Year) or "
            "Certification Name - Issuing Organization (Year).\n\n"
        )
    
    @staticmethod
    def _get_languages_instructions() -> str:
        """Get instructions for Languages section."""
        return (
            "CRITICAL: List ONLY spoken and written communication languages (e.g., English, Spanish, French, Hindi, etc.). "
            "Do NOT include programming languages (e.g., Java, Python, JavaScript, SQL, etc.). "
            "Do NOT include technologies, frameworks, or tools. "
            "For each language, include proficiency level if provided (e.g., Native, Fluent, Professional, Conversational). "
            "Format as: Language Name (Proficiency Level) or just Language Name if no proficiency is specified.\n\n"
        )
    
    @staticmethod
    def _format_candidate_information(
        user_data: Dict,
        exclude_name: bool = False,
        exclude_all_skills: bool = False
    ) -> str:
        """
        Format candidate information for prompts.
        
        Args:
            user_data: User information dictionary
            exclude_name: Whether to exclude name field
            exclude_all_skills: Whether to exclude all_skills field
            
        Returns:
            Formatted candidate information string
        """
        prompt = "Candidate Information:\n"
        
        for key, value in user_data.items():
            if not value:
                continue
            
            # Skip excluded fields
            if exclude_name and key == "name":
                continue
            if exclude_all_skills and key == "all_skills":
                continue
            
            # Format based on value type
            if isinstance(value, list):
                if value:
                    prompt += f"{key}: {', '.join(str(v) for v in value)}\n"
            elif isinstance(value, dict):
                for k, v in value.items():
                    if v:
                        prompt += f"{key} - {k}: {v}\n"
            else:
                prompt += f"{key}: {value}\n"
        
        return prompt
    
    @staticmethod
    def _format_job_requirements(
        job_requirements: Dict,
        include_keywords_only: bool = False
    ) -> str:
        """
        Format job requirements for prompts.
        
        Args:
            job_requirements: Job requirements dictionary
            include_keywords_only: Whether to include only keywords
            
        Returns:
            Formatted job requirements string
        """
        prompt = "\nTarget Job Requirements:\n"
        
        if job_requirements.get("keywords"):
            prompt += f"Key Skills/Keywords: {', '.join(job_requirements['keywords'])}\n"
        
        if not include_keywords_only and job_requirements.get("requirements"):
            prompt += f"Requirements: {', '.join(job_requirements['requirements'][:5])}\n"
        
        if not include_keywords_only:
            prompt += "\nTailor the section to align with these job requirements "
            prompt += "while maintaining accuracy. "
            prompt += "ONLY use information from the Candidate Information above. "
            prompt += "Do NOT invent new experiences or roles.\n"
        
        return prompt
