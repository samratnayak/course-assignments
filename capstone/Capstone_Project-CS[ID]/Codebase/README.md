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

For using Mistral 7B model (recommended for better extraction quality):

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run Mistral 7B model
ollama run mistral

# Start Ollama service (if not already running)
ollama serve

# Verify it's running (in another terminal)
curl http://localhost:11434/api/tags
```

**Note**: 
- Mistral 7B provides better extraction quality than smaller models while remaining fast
- If Ollama is not available, the system will automatically use Flan-T5-XL as the primary model
- The model will be downloaded on first run (may take a few minutes, ~4GB)

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
2. **Model Loading**: Loads Ollama (Mistral 7B) and Flan-T5-XL models for extraction, GPT-4o for optimization
3. **Resume Data Extraction**: Uses LLM to extract structured information
4. **Job Description Parsing** (optional): Parses job requirements for tailoring
5. **CV Generation**: Generates ATS-optimized CV sections using LLMs
6. **Formatting**: Formats and exports CV to specified format

### LLM Models Used

1. **Mistral 7B (via Ollama)**: 
   - High-quality, open-source model
   - Ideal for local, privacy-centric deployment
   - Better extraction quality than smaller models
   - Primary model for LLM 1 (extraction) when available

2. **Flan-T5-XL**:
   - Versatile instruction-following model
   - Used as backup for LLM 1 (extraction)
   - Ensures system works even without Ollama

3. **GPT-4o (OpenAI)**:
   - State-of-the-art optimization model
   - Used for LLM 2 (resume optimization)
   - Best cost-performance ratio for high-quality writing

### Model Justification

- **Mistral 7B**: Chosen for its excellent balance of quality and efficiency, providing better extraction accuracy than smaller models while maintaining fast inference for privacy-sensitive CV generation tasks.
- **Flan-T5-XL**: Selected as a robust backup model with strong instruction-following capabilities, ensuring reliability and consistent output quality when Ollama is unavailable.
- **GPT-4o**: Selected for resume optimization due to its superior writing quality, instruction-following, and cost-effectiveness compared to other GPT-4 variants.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Ollama (optional, for Mistral 7B model)
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
