"""
LangChain Structured Output Parsers (Optional)

This module provides structured output parsing using LangChain's PydanticOutputParser.
Falls back to regex-based parsing if LangChain is not available.

All functions are designed to be drop-in replacements for existing JSON parsing logic.
"""

import json
import re
from typing import Dict, List, Optional, Any

# Try to import LangChain components
LANGCHAIN_AVAILABLE = False
LANGCHAIN_ERROR = None

try:
    # Try different import paths for different LangChain versions
    try:
        from langchain.output_parsers import PydanticOutputParser
    except ImportError:
        try:
            # Newer LangChain versions use langchain_core
            from langchain_core.output_parsers import PydanticOutputParser
        except ImportError:
            try:
                # Alternative path
                from langchain.output_parsers.pydantic import PydanticOutputParser
            except ImportError:
                raise ImportError("Could not find PydanticOutputParser in any LangChain module")
    
    # Try pydantic v2 first, then v1 (for older LangChain versions)
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        try:
            from langchain.pydantic_v1 import BaseModel, Field
        except ImportError:
            try:
                from pydantic.v1 import BaseModel, Field
            except ImportError:
                raise ImportError("Could not find Pydantic BaseModel")
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_ERROR = str(e)
    # Create dummy BaseModel for type hints

# Print status on module load (only once)
if LANGCHAIN_AVAILABLE:
    print("✓ LangChain is available - structured output parsing enabled")
else:
    print(f"ℹ LangChain not available - using regex-based parsing fallback")
    if LANGCHAIN_ERROR:
        print(f"   Error: {LANGCHAIN_ERROR}")
        print(f"   To install: pip install langchain langchain-community pydantic")
    class BaseModel:
        pass


# ============================================================================
# Pydantic Models for Structured Outputs
# ============================================================================

if LANGCHAIN_AVAILABLE:
    class FeedbackInstruction(BaseModel):
        """Single feedback instruction model."""
        section: str = Field(description="CV section name (must match exactly one of the available sections)")
        action: str = Field(description="Action type: add, remove, update, modify")
        feedback: str = Field(description="Specific feedback text for this section")
    
    class FeedbackResponse(BaseModel):
        """Response model for feedback parsing."""
        instructions: List[FeedbackInstruction] = Field(
            default_factory=list,
            description="List of feedback instructions. Empty if stop or all is set."
        )
        stop: Optional[bool] = Field(
            default=False,
            description="True if user wants to stop the feedback loop"
        )
        all: Optional[str] = Field(
            default=None,
            description="Feedback text to apply to all sections if applicable"
        )
    
    class MultiStepInstruction(BaseModel):
        """Model for multi-step instruction parsing."""
        is_multi_step: bool = Field(description="Whether this is a multi-step instruction")
        source_section: Optional[str] = Field(
            default=None,
            description="Source section name (where to remove from)"
        )
        target_section: Optional[str] = Field(
            default=None,
            description="Target section name (where to add to)"
        )
        content_to_move: Optional[str] = Field(
            default=None,
            description="Description of content to move"
        )
        instructions: Dict[str, str] = Field(
            default_factory=dict,
            description="Dictionary mapping section names to feedback instructions"
        )
    
    class ResumeDataModel(BaseModel):
        """Structured model for resume extraction."""
        name: str = Field(description="Candidate's full name (not project names or section headers)")
        skills: List[str] = Field(
            default_factory=list,
            description="Technical skills and programming languages"
        )
        tools: List[str] = Field(
            default_factory=list,
            description="Tools, frameworks, and technologies"
        )
        soft_skills: List[str] = Field(
            default_factory=list,
            description="Soft skills and competencies"
        )
        contact: Dict[str, str] = Field(
            default_factory=lambda: {"email": "", "phone": "", "address": ""},
            description="Contact information"
        )
        education: List[str] = Field(
            default_factory=list,
            description="Education details"
        )
        experience: List[str] = Field(
            default_factory=list,
            description="Work experience summaries"
        )
        projects: List[str] = Field(
            default_factory=list,
            description="Project descriptions"
        )
        certifications: List[str] = Field(
            default_factory=list,
            description="Certifications and credentials"
        )
        languages: List[str] = Field(
            default_factory=list,
            description="Spoken and written languages (not programming languages)"
        )
    
    class JobDescriptionModel(BaseModel):
        """Structured model for job description parsing."""
        job_title: Optional[str] = Field(
            default=None,
            description="Job title or position name"
        )
        skills: List[str] = Field(
            default_factory=list,
            description="Required technical skills"
        )
        tools: List[str] = Field(
            default_factory=list,
            description="Required tools and technologies"
        )
        requirements: List[str] = Field(
            default_factory=list,
            description="Job requirements and qualifications"
        )
        responsibilities: List[str] = Field(
            default_factory=list,
            description="Job responsibilities and duties"
        )


# ============================================================================
# Parsing Functions with LangChain (with fallback)
# ============================================================================

def parse_feedback_response(response: str) -> Dict[str, Any]:
    """
    Parse feedback response using LangChain structured output (if available).
    Falls back to regex-based parsing if LangChain not available.
    
    Args:
        response: LLM response text containing JSON
        
    Returns:
        Dictionary with parsed feedback data
    """
    if not LANGCHAIN_AVAILABLE:
        return _parse_json_fallback(response)
    
    try:
        parser = PydanticOutputParser(pydantic_object=FeedbackResponse)
        
        # Clean response before parsing to handle markdown, extra text, etc.
        cleaned_response = _clean_json_response(response)
        parsed = parser.parse(cleaned_response)
        print("  ✓ Using LangChain structured output parser for feedback parsing")
        
        # Convert to dictionary format expected by existing code
        result = {}
        if parsed.stop:
            result['_stop'] = True
        elif parsed.all:
            result['_all'] = parsed.all
        else:
            result['instructions'] = [
                {
                    'section': inst.section,
                    'action': inst.action,
                    'feedback': inst.feedback
                }
                for inst in parsed.instructions
            ]
        
        return result
        
    except Exception as e:
        # Fallback to regex parsing on any error
        print(f"  ⚠ LangChain parsing failed, falling back to regex: {str(e)[:100]}")
        return _parse_json_fallback(response)


def parse_multi_step_instruction(response: str, current_section: str) -> Dict[str, str]:
    """
    Parse multi-step instruction using LangChain structured output (if available).
    Falls back to regex-based parsing if LangChain not available.
    
    Args:
        response: LLM response text containing JSON
        current_section: Section the feedback was initially routed to
        
    Returns:
        Dictionary mapping section names to feedback instructions
    """
    if not LANGCHAIN_AVAILABLE:
        print("  ℹ LangChain not available, using regex-based parsing for multi-step instruction")
        return _parse_json_fallback(response, default_key=current_section)
    
    try:
        parser = PydanticOutputParser(pydantic_object=MultiStepInstruction)
        # Clean response before parsing to handle markdown, extra text, etc.
        cleaned_response = _clean_json_response(response)
        parsed = parser.parse(cleaned_response)
        print("  ✓ Using LangChain structured output parser for multi-step instruction")
        
        if parsed.is_multi_step and parsed.instructions:
            return parsed.instructions
        else:
            # Not multi-step, return as single instruction
            return {current_section: response}
            
    except Exception as e:
        # Fallback to regex parsing
        print(f"  ⚠ LangChain parsing failed, falling back to regex: {str(e)[:100]}")
        fallback_result = _parse_json_fallback(response)
        if 'instructions' in fallback_result:
            return fallback_result.get('instructions', {current_section: response})
        return {current_section: response}


def parse_resume_data(response: str) -> Dict[str, Any]:
    """
    Parse resume data using LangChain structured output (if available).
    Falls back to regex-based parsing if LangChain not available.
    
    Args:
        response: LLM response text containing JSON
        
    Returns:
        Dictionary with structured resume data
    """
    if not LANGCHAIN_AVAILABLE:
        print("  ℹ LangChain not available, using regex-based parsing for resume extraction")
        return _parse_json_fallback(response)
    
    try:
        parser = PydanticOutputParser(pydantic_object=ResumeDataModel)
        # Clean response before parsing to handle markdown, extra text, etc.
        cleaned_response = _clean_json_response(response)
        parsed = parser.parse(cleaned_response)
        print("  ✓ Using LangChain structured output parser for resume extraction")
        
        # Convert to dictionary format expected by existing code
        return {
            'name': parsed.name,
            'skills': parsed.skills,
            'tools': parsed.tools,
            'soft_skills': parsed.soft_skills,
            'contact': parsed.contact,
            'education': parsed.education,
            'experience': parsed.experience,
            'projects': parsed.projects,
            'certifications': parsed.certifications,
            'languages': parsed.languages
        }
        
    except Exception as e:
        # Fallback to regex parsing - try to extract JSON first
        error_msg = str(e)
        # Show more context if it's a parsing error
        if "Failed to parse" in error_msg:
            # Try to extract the JSON portion from error message
            json_start = response.find('{')
            if json_start != -1:
                json_snippet = response[json_start:json_start+200] + "..."
                print(f"  ⚠ LangChain parsing failed, falling back to regex")
                print(f"     Error: {error_msg[:150]}")
                print(f"     JSON snippet: {json_snippet}")
            else:
                print(f"  ⚠ LangChain parsing failed, falling back to regex: {error_msg[:150]}")
        else:
            print(f"  ⚠ LangChain parsing failed, falling back to regex: {error_msg[:150]}")
        return _parse_json_fallback(response)


def parse_job_description(response: str) -> Dict[str, Any]:
    """
    Parse job description using LangChain structured output (if available).
    Falls back to regex-based parsing if LangChain not available.
    
    Args:
        response: LLM response text containing JSON
        
    Returns:
        Dictionary with structured job description data
    """
    if not LANGCHAIN_AVAILABLE:
        print("  ℹ LangChain not available, using regex-based parsing for job description")
        return _parse_json_fallback(response)
    
    try:
        parser = PydanticOutputParser(pydantic_object=JobDescriptionModel)
        # Clean response before parsing to handle markdown, extra text, etc.
        cleaned_response = _clean_json_response(response)
        parsed = parser.parse(cleaned_response)
        print("  ✓ Using LangChain structured output parser for job description parsing")
        
        # Convert to dictionary format expected by existing code
        return {
            'job_title': parsed.job_title,
            'skills': parsed.skills,
            'tools': parsed.tools,
            'keywords': parsed.skills + parsed.tools,  # For compatibility
            'skills_required': parsed.skills + parsed.tools,  # For compatibility
            'requirements': parsed.requirements,
            'responsibilities': parsed.responsibilities
        }
        
    except Exception as e:
        # Fallback to regex parsing
        print(f"  ⚠ LangChain parsing failed, falling back to regex: {str(e)[:100]}")
        return _parse_json_fallback(response)


# ============================================================================
# Fallback Functions (Current Implementation)
# ============================================================================

def _clean_json_response(response: str) -> str:
    """
    Clean and extract JSON from LLM response.
    Removes markdown code blocks, extra text, and fixes common JSON issues.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks if present
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Try to extract JSON object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # Try to fix common issues: trailing commas, unclosed strings
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing comma before }
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing comma before ]
        return json_str
    
    return response.strip()


def _parse_json_fallback(response: str, default_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fallback JSON parsing using regex (current implementation).
    
    Args:
        response: LLM response text
        default_key: Optional default key if parsing fails
        
    Returns:
        Parsed dictionary or empty dict with default key
    """
    # Clean the response first
    cleaned = _clean_json_response(response)
    
    # Try to parse cleaned JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from response (original method)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to parse the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Return empty dict or dict with default key
    if default_key:
        return {default_key: response}
    return {}


# ============================================================================
# Utility Functions
# ============================================================================

def is_langchain_available() -> bool:
    """Check if LangChain is available."""
    return LANGCHAIN_AVAILABLE


def get_parser_info() -> str:
    """Get information about which parser is being used."""
    if LANGCHAIN_AVAILABLE:
        return "LangChain structured output parser (with Pydantic validation)"
    else:
        return "Regex-based JSON parser (fallback mode)"
