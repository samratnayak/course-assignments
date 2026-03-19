"""
CV Utilities Module

This module contains utility functions for CV generation workflow:
- Display CV previews
- Handle user feedback
- Apply feedback to sections
"""

from typing import Dict, Optional
import json
import re


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


def get_user_feedback(cv_sections: Dict[str, str], cv_generator=None) -> Dict[str, str]:
    """
    Get feedback from user for CV sections using LLM for conversational understanding.
    
    Args:
        cv_sections: Dictionary of current CV sections (for intelligent routing)
        cv_generator: Optional CVGenerator instance for LLM-based parsing
    
    Returns:
        Dictionary mapping section names to feedback strings, or empty dict if no feedback
    """
    print("\n" + "=" * 80)
    print("FEEDBACK OPTIONS")
    print("=" * 80)
    print("You can provide feedback in natural language. Examples:")
    print("- 'Add JavaScript as a skill and add Market Conformity project'")
    print("- 'Remove the last skill and add it to certifications'")
    print("- 'Update my work experience to highlight leadership'")
    print("- Type 'stop', 'done', 'exit', or 'quit' to finish and save the CV")
    print("- Type 'skip' or press Enter to keep current version and continue")
    print("=" * 80)
    
    user_input = input("\nEnter your feedback (or 'stop' to finish): ").strip()
    
    if not user_input or user_input.lower() in ['skip', '']:
        return {}
    
    if user_input.lower() in ['stop', 'done', 'exit', 'quit', 'finish']:
        return {'_stop': True}
    
    # Use LLM to parse feedback conversationally
    if cv_generator:
        feedback_dict = _parse_feedback_with_llm(user_input, cv_sections, cv_generator)
    else:
        # Fallback to simple parsing if LLM not available
        feedback_dict = _parse_feedback_simple(user_input, cv_sections)
    
    return feedback_dict


def _parse_feedback_with_llm(user_input: str, cv_sections: Dict[str, str], cv_generator) -> Dict[str, str]:
    """
    Parse user feedback using LLM for conversational understanding.
    
    Args:
        user_input: User feedback text
        cv_sections: Dictionary of CV sections
        cv_generator: CVGenerator instance with LLM models
    
    Returns:
        Dictionary mapping section names to feedback strings
    """
    # Create prompt for LLM to parse feedback
    sections_list = list(cv_sections.keys())
    
    prompt = f"""You are a helpful assistant that understands user feedback for CV/resume editing.

The CV has the following sections:
{chr(10).join(f"- {section}" for section in sections_list)}

User Feedback:
"{user_input}"

Analyze the user's feedback and determine:
1. What actions they want (add, remove, update, modify, etc.)
2. Which section(s) each instruction applies to
3. The specific content or changes requested
4. If there are multiple instructions, separate them

IMPORTANT:
- If the user mentions "certified", "certification", "certificate", "GCP certified", "AWS certified", etc., 
  it should go to "Certifications" section, NOT "Skills", even if they say "skill"
- If feedback mentions multiple things (e.g., "add X and add Y"), treat them as separate instructions
- If feedback says "remove X from section A and add to section B", create two separate instructions
- Be smart about understanding context (e.g., "project section" means "Projects" section)

Return your response as a JSON object with this exact format:
{{
  "instructions": [
    {{
      "section": "Section Name (must match one of the sections listed above exactly)",
      "action": "add|remove|update|modify",
      "feedback": "The specific feedback text for this section"
    }}
  ]
}}

If the user wants to stop, return: {{"stop": true}}
If the feedback applies to all sections, return: {{"all": "feedback text"}}

JSON Response:"""
    
    try:
        # Use LLM to parse (prefer GPT-4o for better understanding, fallback to primary)
        response = cv_generator.generate_text(
            prompt, 
            use_optimization_model=True,  # Use GPT-4o for better understanding
            temperature=0.3,  # Lower temperature for more consistent parsing
            max_tokens=1000
        )
        
        # Use LangChain structured output parser if available, otherwise fallback to regex
        try:
            from langchain_parsers import parse_feedback_response, is_langchain_available
            if is_langchain_available():
                parsed = parse_feedback_response(response)
            else:
                print("  ℹ LangChain not available, using regex-based parsing for feedback")
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    parsed = json.loads(response)
        except ImportError:
            # Fallback to current regex-based parsing
            print("  ℹ LangChain module not found, using regex-based parsing for feedback")
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response)
        
        # Handle stop signal
        if parsed.get('stop'):
            return {'_stop': True}
        
        # Handle all sections
        if parsed.get('all'):
            return {'_all': parsed['all']}
        
        # Process instructions
        feedback_dict = {}
        if 'instructions' in parsed:
            print(f"\n✓ Understood {len(parsed['instructions'])} instruction(s):")
            for i, instruction in enumerate(parsed['instructions'], 1):
                section = instruction.get('section', '')
                action = instruction.get('action', '')
                feedback_text = instruction.get('feedback', '')
                
                # Validate section name
                if section in cv_sections:
                    feedback_dict[section] = feedback_text
                    print(f"  {i}. {action.capitalize()} in '{section}': {feedback_text[:60]}...")
                else:
                    print(f"  {i}. ⚠ Could not match section '{section}' - please specify manually")
                    # Try to find closest match
                    for sec in cv_sections.keys():
                        if sec.lower() in section.lower() or section.lower() in sec.lower():
                            feedback_dict[sec] = feedback_text
                            print(f"     → Routing to '{sec}' instead")
                            break
        
        return feedback_dict
        
    except json.JSONDecodeError as e:
        print(f"⚠ Could not parse LLM response as JSON: {e}")
        print(f"   LLM Response: {response[:200]}...")
        # Fallback to simple parsing
        return _parse_feedback_simple(user_input, cv_sections)
    except Exception as e:
        print(f"⚠ Error using LLM for feedback parsing: {e}")
        # Fallback to simple parsing
        return _parse_feedback_simple(user_input, cv_sections)


def _parse_feedback_simple(user_input: str, cv_sections: Dict[str, str]) -> Dict[str, str]:
    """
    Simple fallback parsing when LLM is not available.
    
    Args:
        user_input: User feedback text
        cv_sections: Dictionary of CV sections
    
    Returns:
        Dictionary mapping section names to feedback strings
    """
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
    feedback_dict: Dict[str, str],
    cv_generator=None
) -> Dict[str, str]:
    """
    Determine which sections need to be updated based on feedback.
    Uses LLM to understand multi-step instructions if available.
    
    Args:
        cv_sections: Current CV sections
        feedback_dict: Dictionary mapping section names to feedback
        cv_generator: Optional CVGenerator instance for LLM-based understanding
        
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
        # Process each feedback entry
        for section_name, feedback in feedback_dict.items():
            if section_name not in cv_sections:
                continue
            
            # Use LLM to understand if this is a multi-step instruction
            if cv_generator and _is_multi_step_instruction(feedback):
                multi_step_result = _parse_multi_step_with_llm(
                    feedback, section_name, cv_sections, cv_generator
                )
                sections_to_update.update(multi_step_result)
            else:
                sections_to_update[section_name] = feedback
    
    return sections_to_update


def _is_multi_step_instruction(feedback: str) -> bool:
    """
    Check if feedback contains multi-step instructions (remove from X, add to Y).
    
    Args:
        feedback: Feedback text
        
    Returns:
        True if it appears to be a multi-step instruction
    """
    feedback_lower = feedback.lower()
    return ('remove' in feedback_lower and 
            ('add' in feedback_lower or 'move' in feedback_lower or 'transfer' in feedback_lower))


def _parse_multi_step_with_llm(
    feedback: str, 
    current_section: str, 
    cv_sections: Dict[str, str], 
    cv_generator
) -> Dict[str, str]:
    """
    Use LLM to parse multi-step instructions like "remove X from section A and add to section B".
    
    Args:
        feedback: User feedback text
        current_section: Section the feedback was initially routed to
        cv_sections: Dictionary of CV sections
        cv_generator: CVGenerator instance with LLM models
        
    Returns:
        Dictionary of sections to update with their feedback
    """
    sections_list = list(cv_sections.keys())
    
    prompt = f"""You are analyzing user feedback for CV editing that involves moving content between sections.

Current CV Sections:
{chr(10).join(f"- {section}" for section in sections_list)}

User Feedback:
"{feedback}"

This feedback was initially routed to: "{current_section}"

Analyze if this feedback involves:
1. Removing something from one section
2. Adding it to another section
3. Or any other multi-step operation

If it's a multi-step instruction (e.g., "remove X and add to Y"), identify:
- Source section (where to remove from)
- Target section (where to add to)
- What content to move

Return JSON in this format:
{{
  "is_multi_step": true/false,
  "source_section": "Section name (if removing)",
  "target_section": "Section name (if adding)",
  "content_to_move": "Description of what to move",
  "instructions": {{
    "source_section_name": "Feedback for source section (remove instruction)",
    "target_section_name": "Feedback for target section (add instruction)"
  }}
}}

If it's NOT multi-step, return:
{{
  "is_multi_step": false,
  "instructions": {{
    "{current_section}": "{feedback}"
  }}
}}

JSON Response:"""
    
    try:
        response = cv_generator.generate_text(
            prompt,
            use_optimization_model=True,
            temperature=0.3,
            max_tokens=500
        )
        
        # Use LangChain structured output parser if available, otherwise fallback to regex
        try:
            from langchain_parsers import parse_multi_step_instruction
            instructions = parse_multi_step_instruction(response, current_section)
            if instructions and isinstance(instructions, dict) and len(instructions) > 1:
                print(f"  ✓ Multi-step instruction detected:")
                for sec, instr in instructions.items():
                    if sec in cv_sections:
                        print(f"    - {sec}: {instr[:50]}...")
                return instructions
            else:
                # Not multi-step, return as single instruction
                return {current_section: feedback}
        except ImportError:
            # Fallback to current regex-based parsing
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if parsed.get('is_multi_step') and 'instructions' in parsed:
                        instructions = parsed['instructions']
                        print(f"  ✓ Multi-step instruction detected:")
                        for sec, instr in instructions.items():
                            if sec in cv_sections:
                                print(f"    - {sec}: {instr[:50]}...")
                        return instructions
                except json.JSONDecodeError:
                    pass
            
            # Not multi-step, return as single instruction
            return {current_section: feedback}
            
    except Exception as e:
        print(f"  ⚠ Could not parse multi-step instruction: {e}")
        return {current_section: feedback}
