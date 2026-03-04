"""
CV Formatter Module

This module handles formatting and exporting CVs to DOCX and PDF formats.
"""

import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class CVFormatter:
    """
    Class to format and export CVs to DOCX and PDF formats.
    """
    
    def __init__(self):
        """Initialize the CVFormatter."""
        pass
    
    def format_to_docx(self, cv_sections: Dict[str, str], output_path: str, candidate_name: str = "Candidate"):
        """
        Format CV sections into a DOCX document.
        
        Args:
            cv_sections: Dictionary of CV sections
            output_path: Path to save the DOCX file
            candidate_name: Name of the candidate for header
        """
        # Create a new Document
        doc = Document()
        
        # Set document margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
        
        # Add title/header
        title = doc.add_heading(candidate_name, 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add sections
        for section_name, section_content in cv_sections.items():
            # Add section heading
            heading = doc.add_heading(section_name, level=1)
            heading_format = heading.paragraph_format
            heading_format.space_before = Pt(12)
            heading_format.space_after = Pt(6)
            
            # Add section content
            paragraphs = section_content.split('\n')
            for para_text in paragraphs:
                if para_text.strip():
                    para = doc.add_paragraph(para_text.strip())
                    para_format = para.paragraph_format
                    para_format.space_after = Pt(6)
            
            # Add spacing between sections
            doc.add_paragraph()
        
        # Save document
        doc.save(output_path)
        print(f"CV saved to: {output_path}")
    
    def format_to_text(self, cv_sections: Dict[str, str], output_path: str):
        """
        Format CV sections into a plain text file.
        
        Args:
            cv_sections: Dictionary of CV sections
            output_path: Path to save the text file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for section_name, section_content in cv_sections.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"{section_name.upper()}\n")
                f.write(f"{'='*80}\n")
                f.write(f"{section_content}\n\n")
        
        print(f"CV saved to: {output_path}")
    
    def format_to_pdf(self, cv_sections: Dict[str, str], output_path: str, candidate_name: str = "Candidate"):
        """
        Format CV sections into a PDF document.
        First creates DOCX, then converts to PDF (requires external tool or library).
        
        Args:
            cv_sections: Dictionary of CV sections
            output_path: Path to save the PDF file
            candidate_name: Name of the candidate
        """
        # For now, create DOCX first
        # PDF conversion would require additional libraries like pdfkit or reportlab
        # Or use pandoc if available: pandoc input.docx -o output.pdf
        
        docx_path = output_path.replace('.pdf', '.docx')
        self.format_to_docx(cv_sections, docx_path, candidate_name)
        
        # Try to convert to PDF using pandoc if available
        try:
            import subprocess
            result = subprocess.run(
                ['pandoc', docx_path, '-o', output_path],
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"CV saved to PDF: {output_path}")
                # Optionally remove DOCX file
                # os.remove(docx_path)
            else:
                print(f"PDF conversion failed. DOCX saved at: {docx_path}")
                print("Note: Install pandoc for PDF conversion, or use the DOCX file.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"PDF conversion not available. DOCX saved at: {docx_path}")
            print("Note: Install pandoc (https://pandoc.org/installing.html) for PDF conversion.")
    
    def format_cv(self, cv_sections: Dict[str, str], output_path: str, 
                  format_type: str = "docx", candidate_name: str = "Candidate"):
        """
        Format CV to specified format.
        
        Args:
            cv_sections: Dictionary of CV sections
            output_path: Path to save the file
            format_type: Output format ('docx', 'txt', 'pdf')
            candidate_name: Name of the candidate
        """
        if format_type.lower() == "docx":
            self.format_to_docx(cv_sections, output_path, candidate_name)
        elif format_type.lower() == "txt":
            self.format_to_text(cv_sections, output_path)
        elif format_type.lower() == "pdf":
            self.format_to_pdf(cv_sections, output_path, candidate_name)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
