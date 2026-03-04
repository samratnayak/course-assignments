# CV Creation using LLMs - Capstone Project (CS[01])

## Project Overview

This project implements an automated CV/Resume creation system using Large Language Models (LLMs). The system takes unstructured user profile data (from text files, PDFs, or Word documents) as input and generates professional, ATS-friendly CVs using multiple LLM models.

## Features

- **Multiple LLM Support**: Uses at least 2 LLMs:
  - **Gemma 3 1B via Ollama** (primary model, if available)
  - **Flan-T5-XL** (secondary/backup model)
- **Document Processing**: Extracts text from PDF, DOCX, and TXT files
- **Resume Data Extraction**: Uses LLMs to extract structured information from unstructured data
- **Job Description Parsing**: Optional job description parsing for CV tailoring
- **ATS-Optimized Generation**: Generates keyword-rich, ATS-friendly CV sections
- **Multiple Output Formats**: Supports DOCX, TXT, and PDF output formats

## Project Structure

```
Codebase/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── document_extractor.py   # PDF/Word document text extraction
├── resume_extractor.py     # LLM-based resume data extraction
├── job_parser.py           # Job description parsing
├── cv_generator.py         # CV generation using multiple LLMs
├── cv_formatter.py         # CV formatting and export (DOCX/PDF)
├── requirements.txt        # Python dependencies
├── execution.txt           # Execution command
├── input/                  # Input files directory
│   └── sample_profile.txt  # Sample input file
└── output/                 # Generated CVs directory
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd Codebase
pip install -r requirements.txt
```

### 2. Setup Ollama (Optional but Recommended)

For using Gemma 3 1B model:

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/

# Pull Gemma 2 1B model
ollama pull gemma2:1b

# Start Ollama service (usually runs automatically)
# Verify: curl http://localhost:11434/api/tags
```

**Note**: If Ollama is not available, the system will automatically use Flan-T5-XL as the primary model.

### 3. Run the Application

```bash
# Basic usage
python main.py input/sample_profile.txt

# With custom output path and format
python main.py input/sample_profile.txt -o output/my_cv.docx -f docx

# With job description for tailored CV
python main.py input/sample_profile.txt -j job_description.txt -o output/tailored_cv.docx

# Using execution.txt
bash execution.txt
```

## Usage Examples

### Example 1: Basic CV Generation

```bash
python main.py input/sample_profile.txt -o output/cv_output.docx
```

### Example 2: Generate PDF CV

```bash
python main.py input/sample_profile.txt -f pdf -o output/cv_output.pdf
```

### Example 3: Tailored CV with Job Description

```bash
python main.py input/sample_profile.txt -j job_description.txt -o output/tailored_cv.docx
```

### Example 4: Process PDF Input

```bash
python main.py input/resume.pdf -o output/cv_from_pdf.docx
```

## Command Line Arguments

- `input_file` (required): Path to input file (PDF, DOCX, or TXT)
- `-o, --output`: Output file path (default: `output/cv_[name].[format]`)
- `-f, --format`: Output format - `docx`, `txt`, or `pdf` (default: `docx`)
- `-j, --job-description`: Optional path to job description file for CV tailoring
- `--api-key`: Optional API key (can also be set via environment variable)

## Implementation Details

### Step-by-Step Process

1. **Document Extraction**: Extracts text from input files (PDF/Word/TXT)
2. **Model Loading**: Loads Ollama (Gemma) and Flan-T5-XL models
3. **Resume Data Extraction**: Uses LLM to extract structured information
4. **Job Description Parsing** (optional): Parses job requirements for tailoring
5. **CV Generation**: Generates ATS-optimized CV sections using LLMs
6. **Formatting**: Formats and exports CV to specified format

### LLM Models Used

1. **Gemma 3 1B (via Ollama)**: 
   - Lightweight, open-source model
   - Ideal for local, privacy-centric deployment
   - Primary model when available

2. **Flan-T5-XL**:
   - Versatile instruction-following model
   - Used as secondary/backup model
   - Ensures system works even without Ollama

### Model Justification

- **Gemma 3 1B**: Chosen for its efficiency and local deployment capability, making it ideal for privacy-sensitive CV generation tasks.
- **Flan-T5-XL**: Selected as a robust backup model with strong instruction-following capabilities, ensuring reliability and consistent output quality.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Ollama (optional, for Gemma model)
- CUDA (optional, for GPU acceleration)

## Notes

- The system automatically falls back to Flan-T5-XL if Ollama is not available
- Models will be downloaded on first run (requires internet connection)
- For GPU usage, ensure CUDA is properly installed
- PDF output requires pandoc (optional, DOCX is always available)

## Project Compliance

- ✅ Uses at least 2 LLMs (Gemma via Ollama + Flan-T5-XL)
- ✅ All code files in Codebase directory
- ✅ main.py as entry point
- ✅ execution.txt with execution command
- ✅ Proper comments throughout code
- ✅ Terminal-executable (no GUI required)
- ✅ Input: Unstructured user profile data
- ✅ Output: Final CVs in DOCX/PDF format

## Author

[Your Name/ID]

## License

[Specify license if applicable]
