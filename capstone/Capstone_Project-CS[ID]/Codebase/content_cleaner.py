"""
Content Cleaning Module

This module handles cleaning and sanitization of LLM-generated CV content.
Removes section titles, markdown formatting, quotes, and placeholder text.
"""

import re
from typing import List


class ContentCleaner:
    """
    Handles cleaning of generated CV content.
    
    Methods:
        clean_section_content: Main method to clean CV section content
    """
    
    # Placeholder patterns for Personal Information section
    PLACEHOLDER_PATTERNS = [
        r'\[linkedin\s+url\s+if\s+provided\]',
        r'\[address\s+if\s+provided\]',
        r'\[linkedin\s+url\s+if\s+available\]',
        r'\[address\s+if\s+available\]',
        r'linkedin\s+url\s+if\s+provided',
        r'address\s+if\s+provided',
        r'linkedin\s+url\s+if\s+available',
        r'address\s+if\s+available',
    ]
    
    @staticmethod
    def clean_section_content(content: str, section_name: str) -> str:
        """
        Clean generated content to remove section titles, markdown, and placeholders.
        
        Args:
            content: Generated content that may contain section titles
            section_name: Name of the section
            
        Returns:
            Cleaned content without section titles, markdown, or placeholders
        """
        if not content:
            return content
        
        # Step 1: Remove section titles and separators
        content = ContentCleaner._remove_section_titles(content, section_name)
        
        # Step 2: Special cleaning for Personal Information
        if section_name == "Personal Information":
            content = ContentCleaner._clean_personal_information(content)
        
        # Step 2b: Special cleaning for Languages section
        if section_name == "Languages":
            content = ContentCleaner._clean_languages_section(content)
        
        # Step 3: Remove markdown and quotes
        content = ContentCleaner._remove_markdown_and_quotes(content)
        
        return content.strip()
    
    @staticmethod
    def _remove_section_titles(content: str, section_name: str) -> str:
        """
        Remove section titles and separators from content.
        
        Args:
            content: Content to clean
            section_name: Name of the section
            
        Returns:
            Content with titles removed
        """
        lines = content.split('\n')
        cleaned_lines = []
        
        # Common patterns for section titles
        section_patterns = [
            section_name,
            section_name.upper(),
            section_name.title(),
            f"**{section_name}**",
            f"**{section_name.upper()}**",
            f"**{section_name.title()}**",
            f"#{section_name}",
            f"##{section_name}",
            f"###{section_name}",
        ]
        
        skip_next = False
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines at the start
            if not cleaned_lines and not line_stripped:
                continue
            
            # Remove quotes from line
            original_line = ContentCleaner._remove_quotes_from_line(line)
            line_stripped = original_line.strip()
            
            # Check if this line is a section title
            is_title = ContentCleaner._is_section_title(line_stripped, section_patterns)
            
            if is_title:
                skip_next = True
                continue
            
            # Skip separator lines after titles
            if skip_next and ContentCleaner._is_separator_line(line_stripped):
                skip_next = False
                continue
            
            skip_next = False
            cleaned_lines.append(original_line)
        
        cleaned_content = '\n'.join(cleaned_lines).strip()
        
        # Remove any remaining title patterns at the start
        for pattern in section_patterns:
            if cleaned_content.startswith(pattern):
                cleaned_content = cleaned_content[len(pattern):].strip()
                cleaned_content = cleaned_content.lstrip(': -').strip()
        
        return cleaned_content
    
    @staticmethod
    def _clean_personal_information(content: str) -> str:
        """
        Clean Personal Information section - remove Contact Information and placeholders.
        
        Args:
            content: Content to clean
            
        Returns:
            Cleaned content
        """
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            line_lower = line_stripped.lower()
            
            # Skip "Contact Information" lines
            if ContentCleaner._is_contact_information_line(line_lower):
                continue
            
            # Skip placeholder lines
            if ContentCleaner._is_placeholder_line(line_stripped, line_lower):
                continue
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines).strip()
    
    @staticmethod
    def _remove_markdown_and_quotes(content: str) -> str:
        """
        Remove markdown formatting and quotes from content.
        
        Args:
            content: Content to clean
            
        Returns:
            Cleaned content
        """
        final_lines = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                final_lines.append('')
                continue
            
            # Remove markdown formatting
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)  # Remove **bold**
            line = re.sub(r'\*([^*]+)\*', r'\1', line)  # Remove *italic*
            
            # Remove quotes
            line = ContentCleaner._remove_quotes_from_line(line)
            
            final_lines.append(line)
        
        cleaned_content = '\n'.join(final_lines).strip()
        
        # Remove quotes from entire content if wrapped
        if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
            cleaned_content = cleaned_content[1:-1].strip()
        elif cleaned_content.startswith("'") and cleaned_content.endswith("'"):
            cleaned_content = cleaned_content[1:-1].strip()
        
        # Remove markdown from entire content
        cleaned_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_content)
        cleaned_content = re.sub(r'\*([^*]+)\*', r'\1', cleaned_content)
        
        return cleaned_content
    
    @staticmethod
    def _remove_quotes_from_line(line: str) -> str:
        """
        Remove quotes from a single line.
        
        Args:
            line: Line to clean
            
        Returns:
            Line with quotes removed
        """
        line_stripped = line.strip()
        
        # Handle quotes wrapping entire line
        if line_stripped.startswith('"') and line_stripped.endswith('"'):
            return line_stripped[1:-1].strip()
        elif line_stripped.startswith("'") and line_stripped.endswith("'"):
            return line_stripped[1:-1].strip()
        
        # Remove quotes from start/end individually
        return line_stripped.strip('"').strip("'").strip()
    
    @staticmethod
    def _is_section_title(line: str, patterns: List[str]) -> bool:
        """
        Check if a line is a section title.
        
        Args:
            line: Line to check
            patterns: List of section title patterns
            
        Returns:
            True if line is a section title
        """
        for pattern in patterns:
            if pattern in line and len(line) < 100:
                return True
            if line == pattern:
                return True
        return False
    
    @staticmethod
    def _is_separator_line(line: str) -> bool:
        """
        Check if a line is a separator (dashes, equals, etc.).
        
        Args:
            line: Line to check
            
        Returns:
            True if line is a separator
        """
        if line.startswith('-') * 10 or line.startswith('=') * 10:
            return True
        if all(c in '-=_' for c in line) and len(line) > 10:
            return True
        return False
    
    @staticmethod
    def _is_contact_information_line(line_lower: str) -> bool:
        """
        Check if a line is "Contact Information" heading.
        
        Args:
            line_lower: Lowercase line to check
            
        Returns:
            True if line is Contact Information heading
        """
        if line_lower in ['contact information', 'contact information:', 'contact information :']:
            return True
        if line_lower.startswith('contact information'):
            remaining = line_lower.replace('contact information', '').replace(':', '').strip()
            if len(remaining) < 5:
                return True
        return False
    
    @staticmethod
    def _is_placeholder_line(line_stripped: str, line_lower: str) -> bool:
        """
        Check if a line is a placeholder.
        
        Args:
            line_stripped: Stripped line
            line_lower: Lowercase line
            
        Returns:
            True if line is a placeholder
        """
        # Check regex patterns
        for pattern in ContentCleaner.PLACEHOLDER_PATTERNS:
            if re.search(pattern, line_lower):
                return True
        
        # Check bracket-wrapped placeholders
        if line_stripped.startswith('[') and line_stripped.endswith(']'):
            if any(keyword in line_lower for keyword in ['if provided', 'if available', 'linkedin', 'address']):
                return True
        
        return False
    
    @staticmethod
    def _clean_languages_section(content: str) -> str:
        """
        Clean Languages section - remove programming languages and technologies.
        
        Args:
            content: Content to clean
            
        Returns:
            Cleaned content with only spoken/written languages
        """
        import re
        
        # Common programming languages and technologies to remove
        programming_languages = [
            r'\bjava\b', r'\bpython\b', r'\bjavascript\b', r'\btypescript\b',
            r'\bc\+\+\b', r'\bc#\b', r'\bgo\b', r'\brust\b', r'\bkotlin\b',
            r'\bswift\b', r'\bphp\b', r'\bruby\b', r'\bperl\b', r'\bscala\b',
            r'\bsql\b', r'\bhtml\b', r'\bcss\b', r'\bxml\b', r'\bjson\b',
            r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bnode\.js\b',
            r'\bspring\b', r'\bspringboot\b', r'\bdjango\b', r'\bflask\b',
            r'\bspark\b', r'\bhadoop\b', r'\bkafka\b', r'\bcassandra\b',
            r'\bredis\b', r'\bmongodb\b', r'\bmysql\b', r'\bpostgresql\b',
            r'\bdocker\b', r'\bkubernetes\b', r'\baws\b', r'\bazure\b',
            r'\bgcp\b', r'\bterraform\b', r'\bansible\b', r'\bjenkins\b',
            r'\bgit\b', r'\bgithub\b', r'\bgitlab\b', r'\bjira\b',
            r'\bmachine learning\b', r'\bdeep learning\b', r'\bai\b',
            r'\bdata science\b', r'\bprompt engineering\b', r'\bopenai\b',
            r'\bllm\b', r'\bapi\b', r'\brest\b', r'\bmicroservices\b',
            r'\bcloud\b', r'\bdevops\b', r'\bci/cd\b', r'\bagile\b',
            r'\bscrum\b', r'\bcontainers\b'
        ]
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower()
            # Check if line contains programming language keywords
            is_programming_lang = False
            for pattern in programming_languages:
                if re.search(pattern, line_lower):
                    is_programming_lang = True
                    break
            
            # Skip lines that are clearly about programming/technologies
            if is_programming_lang:
                continue
            
            # Also skip lines that mention "programming", "technology", "framework", etc.
            skip_keywords = [
                'programming', 'technology', 'framework', 'library', 'tool',
                'platform', 'database', 'software', 'application', 'system',
                'development', 'coding', 'scripting', 'experience with',
                'proficient in', 'familiar with', 'extensive experience',
                'building', 'developing', 'using', 'technologies such as'
            ]
            
            if any(keyword in line_lower for keyword in skip_keywords):
                # Check if it's actually about a spoken language (e.g., "English for professional communication")
                spoken_lang_keywords = [
                    'english', 'spanish', 'french', 'german', 'hindi',
                    'chinese', 'japanese', 'korean', 'arabic', 'portuguese',
                    'italian', 'russian', 'bengali', 'tamil', 'telugu',
                    'marathi', 'gujarati', 'punjabi', 'urdu', 'oriya',
                    'malayalam', 'kannada', 'communication', 'spoken',
                    'written', 'fluent', 'native', 'bilingual', 'professional'
                ]
                if not any(lang in line_lower for lang in spoken_lang_keywords):
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()