#!/usr/bin/env python3
"""Temporary script to extract text from PDF files."""
import sys
try:
    import PyPDF2
    def read_pdf(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
except ImportError:
    try:
        import pdfplumber
        def read_pdf(pdf_path):
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
    except ImportError:
        print("No PDF library available. Please install PyPDF2 or pdfplumber.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pdf.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    try:
        text = read_pdf(pdf_path)
        print(text)
    except Exception as e:
        print(f"Error reading PDF: {e}", file=sys.stderr)
        sys.exit(1)
