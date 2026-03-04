# CV Creation using LLMs - Capstone Project

## Project Overview

This project implements an automated CV/Resume creation system using Large Language Models (LLMs). The system takes user information as input and generates professional CV sections using advanced language models.

## Project Structure

```
Capstone_Project-CS[ID]/
├── Codebase/
│   ├── main.py              # Main entry point
│   ├── cv_generator.py      # CV generation logic
│   ├── config.py            # Configuration settings
│   ├── requirements.txt      # Python dependencies
│   ├── execution.txt        # Execution command
│   ├── input/               # Input files directory
│   └── output/              # Generated CVs directory
└── Report/
    ├── Report.pdf           # Project report
    └── Report.docx          # Project report (Word format)
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd Codebase
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

Or use the execution command:
```bash
bash execution.txt
```

## Features

- **Automated CV Generation**: Generate professional CV sections using LLMs
- **Multiple Sections**: Support for various CV sections (Personal Info, Experience, Education, Skills, etc.)
- **Customizable Templates**: Configurable CV structure and formatting
- **Model Flexibility**: Easy to switch between different LLM models

## Configuration

Edit `config.py` to customize:
- Model selection
- Generation parameters (temperature, top_p, etc.)
- CV section structure
- Input/output paths

## Usage

1. Prepare your input data (user information)
2. Configure the model and generation parameters in `config.py`
3. Run `main.py` to generate the CV
4. Check the `output/` directory for generated CVs

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- CUDA (optional, for GPU acceleration)

## Notes

- The project uses the Flan-T5-XL model by default
- Model will be downloaded on first run (requires internet connection)
- For GPU usage, ensure CUDA is properly installed

## Author

[Your Name/ID]

## License

[Specify license if applicable]
