"""
CV Utilities Module

This module contains utility functions for CV generation workflow:
- Display CV previews
- Handle user feedback
- Apply feedback to sections
"""

from typing import Dict, Optional


def display_cv_preview(cv_sections: Dict[str, str], iteration: int = 1) -> None:
    """
    Display a preview of the CV sections for user review.
    
    Args:
        cv_sections: Dictionary of CV sections
        iteration: Current iteration number
    """
    print("\n" + "=" * 80)
    print(f"CV PREVIEW (Iteration {iteration})")
    print("=" * 80)
    
    for section_name, section_content in cv_sections.items():
        print(f"\n[{section_name}]")
        print("-" * 80)
        # Display first 500 characters of each section to keep it manageable
        preview = section_content[:500] if len(section_content) > 500 else section_content
        print(preview)
        if len(section_content) > 500:
            print("... (truncated)")
    
    print("\n" + "=" * 80)


def get_user_feedback(cv_sections: Dict[str, str]) -> Dict[str, str]:
    """
    Get feedback from user for CV sections.
    
    Args:
        cv_sections: Dictionary of current CV sections (for intelligent routing)
    
    Returns:
        Dictionary mapping section names to feedback strings, or empty dict if no feedback
    """
    print("\n" + "=" * 80)
    print("FEEDBACK OPTIONS")
    print("=" * 80)
    print("You can provide feedback in the following ways:")
    print("1. Type 'stop', 'done', 'exit', or 'quit' to finish and save the CV")
    print("2. Type 'all: [your feedback]' to provide feedback for all sections")
    print("3. Type '[section name]: [your feedback]' to provide feedback for a specific section")
    print("4. Type 'skip' or press Enter to keep current version and continue")
    print("=" * 80)
    
    user_input = input("\nEnter your feedback (or 'stop' to finish): ").strip()
    
    if not user_input or user_input.lower() in ['skip', '']:
        return {}
    
    if user_input.lower() in ['stop', 'done', 'exit', 'quit', 'finish']:
        return {'_stop': True}
    
    feedback_dict = {}
    
    # Check if feedback is for all sections
    if user_input.lower().startswith('all:'):
        feedback_text = user_input[4:].strip()
        if feedback_text:
            feedback_dict['_all'] = feedback_text
    else:
        # Check if feedback specifies a section
        parts = user_input.split(':', 1)
        if len(parts) == 2:
            section_name = parts[0].strip()
            feedback_text = parts[1].strip()
            if feedback_text:
                feedback_dict[section_name] = feedback_text
        else:
            # Try to detect relevant section automatically
            relevant_section = _detect_relevant_section(user_input, cv_sections)
            if relevant_section:
                feedback_dict[relevant_section] = user_input
                print(f"✓ Feedback will be applied to '{relevant_section}' section")
            else:
                # Ask user to specify section
                feedback_dict = _prompt_for_section(user_input, cv_sections)
    
    return feedback_dict


def _detect_relevant_section(feedback: str, cv_sections: Dict[str, str]) -> Optional[str]:
    """
    Intelligently detect which section the feedback is most relevant to.
    
    Args:
        feedback: User feedback text
        cv_sections: Dictionary of CV sections
        
    Returns:
        Section name if detected, None otherwise
    """
    feedback_lower = feedback.lower()
    
    # Keywords that indicate specific sections
    section_keywords = {
        "Personal Information": [
            "availability", "notice period", "notice", "joining", "start date",
            "contact", "email", "phone", "address", "location", "relocation",
            "visa", "citizenship"
        ],
        "Professional Summary": [
            "summary", "overview", "profile", "objective", "about me"
        ],
        "Work Experience": [
            "experience", "job", "work", "employment", "position", "role",
            "company", "employer", "career"
        ],
        "Education": [
            "education", "degree", "university", "college", "school", "gpa",
            "graduation", "diploma", "certificate"
        ],
        "Skills": [
            "skill", "technical", "proficiency", "competency", "ability", "expertise"
        ],
        "Certifications": [
            "certification", "certified", "certificate", "license", "credential"
        ],
        "Projects": [
            "project", "portfolio", "work sample"
        ],
        "Languages": [
            "language", "fluent", "bilingual", "speaking"
        ]
    }
    
    # Score each section based on keyword matches
    section_scores = {}
    for section, keywords in section_keywords.items():
        score = sum(1 for keyword in keywords if keyword in feedback_lower)
        if score > 0:
            section_scores[section] = score
    
    # Return section with highest score
    if section_scores:
        return max(section_scores, key=section_scores.get)
    
    return None


def _prompt_for_section(feedback: str, cv_sections: Dict[str, str]) -> Dict[str, str]:
    """
    Prompt user to select which section the feedback applies to.
    
    Args:
        feedback: User feedback text
        cv_sections: Dictionary of CV sections
        
    Returns:
        Dictionary with section name and feedback
    """
    print("\nWhich section should this feedback apply to?")
    print("Available sections:")
    for i, section in enumerate(cv_sections.keys(), 1):
        print(f"  {i}. {section}")
    print("  Or type 'all' to apply to all sections")
    
    section_choice = input("\nEnter section number or name (or 'all'): ").strip()
    
    if section_choice.lower() == 'all':
        return {'_all': feedback}
    
    # Try to match section
    matched_section = None
    
    # Try as number
    try:
        section_num = int(section_choice)
        sections_list = list(cv_sections.keys())
        if 1 <= section_num <= len(sections_list):
            matched_section = sections_list[section_num - 1]
    except ValueError:
        # Try as name
        for section in cv_sections.keys():
            if (section_choice.lower() in section.lower() or
                    section.lower() in section_choice.lower()):
                matched_section = section
                break
    
    if matched_section:
        return {matched_section: feedback}
    else:
        print(f"⚠ Could not match '{section_choice}'. Applying to all sections.")
        return {'_all': feedback}


def apply_feedback_to_sections(
    cv_sections: Dict[str, str],
    feedback_dict: Dict[str, str]
) -> Dict[str, str]:
    """
    Determine which sections need to be updated based on feedback.
    
    Args:
        cv_sections: Current CV sections
        feedback_dict: Dictionary mapping section names to feedback
        
    Returns:
        Dictionary of sections that need to be regenerated
    """
    sections_to_update = {}
    
    # Check for stop signal
    if feedback_dict.get('_stop'):
        return {}
    
    # Check for feedback to all sections
    if '_all' in feedback_dict:
        sections_to_update = {section: feedback_dict['_all'] 
                             for section in cv_sections.keys()}
    else:
        # Apply feedback to specific sections
        for section_name, feedback in feedback_dict.items():
            if section_name in cv_sections:
                sections_to_update[section_name] = feedback
    
    return sections_to_update
