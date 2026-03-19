"""
Document Extractor Module

This module handles extraction of text from various document formats (PDF, Word, etc.)
and converts them to structured text or JSON format.
"""

import os
import json
import pdfplumber
from docx import Document
from typing import Dict, Optional, Union
import warnings

warnings.filterwarnings("ignore")


class DocumentExtractor:
    """
    Class to extract text from PDF and Word documents.
    """
    
    def __init__(self):
        """Initialize the DocumentExtractor."""
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
    
    def extract_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            raise
        return text.strip()
    
    def extract_from_docx(self, file_path: str) -> str:
        """
        Extract text from a Word document (.docx).
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error extracting from DOCX: {e}")
            raise
        return text.strip()
    
    def extract_from_txt(self, file_path: str) -> str:
        """
        Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error extracting from TXT: {e}")
            raise
        return text.strip()
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        elif file_ext == '.txt':
            return self.extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def extract_from_multiple_files(self, file_paths: list) -> str:
        """
        Extract text from multiple files and combine them.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Combined extracted text as string
        """
        all_texts = []
        for file_path in file_paths:
            try:
                text = self.extract_text(file_path)
                all_texts.append(text)
            except Exception as e:
                print(f"Warning: Failed to extract from {file_path}: {e}")
                continue
        
        if not all_texts:
            raise ValueError("No text could be extracted from any of the provided files")
        
        # Combine texts with separator
        return "\n\n".join(all_texts)
    
    def extract_to_json(self, file_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Extract text from document and save as JSON.
        
        Args:
            file_path: Path to the input document
            output_path: Optional path to save JSON file
            
        Returns:
            Dictionary with extracted text
        """
        text = self.extract_text(file_path)
        data = {
            "source_file": file_path,
            "extracted_text": text,
            "file_type": os.path.splitext(file_path)[1].lower()
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return data
