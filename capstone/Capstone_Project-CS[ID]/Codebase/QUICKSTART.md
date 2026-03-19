# Quick Start Guide - CV Creation using LLMs

## Step 1: Install Dependencies

Navigate to the Codebase directory and install required packages:

```bash
cd "Capstone_Project-CS[ID]/Codebase"
pip install -r requirements.txt
```

**Note**: This will install PyTorch, Transformers, OpenAI, SentenceTransformers, and other required libraries. The first run may take some time as models are downloaded.

## Step 2: Setup OpenAI API Key (Required for LLM 2 - GPT-4o)

The project uses GPT-4o for resume optimization. Set your OpenAI API key:

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Command Line Argument**
```bash
python main.py input/sample_profile.txt --api-key "your-api-key-here"
```

## Step 3: (Optional) Setup Ollama for Mistral 7B Model

For faster and better extraction, you can use Ollama with Mistral 7B:

1. **Install Ollama**: Visit https://ollama.ai/ and install Ollama, or run:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull and run Mistral 7B model**:
   ```bash
   ollama run mistral
   ```

3. **Start Ollama service** (if not already running):
   ```bash
   ollama serve
   ```

4. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

**Note**: 
- Mistral 7B provides better extraction quality than smaller models while remaining fast
- If Ollama is not available, the system will automatically use Flan-T5-XL as the primary model
- The model will be downloaded on first run (may take a few minutes, ~4GB)

## Step 4: Run the Application

### Basic Usage (No Job Description)

```bash
python main.py input/sample_profile.txt
```

This will:
- Extract text from the input file
- Load LLM models (Mistral 7B via Ollama if available, otherwise Flan-T5-XL for extraction; GPT-4o for optimization)
- Extract structured resume data
- Generate CV sections
- Enter conversational refinement mode for iterative improvements
- Save output to `output/cv_[name].docx`

### With Job Description for ATS-Optimized CV

```bash
python main.py input/sample_profile.txt -j "Job description text here" -o output/tailored_cv.docx
```

Or with a job description file:
```bash
python main.py input/sample_profile.txt -j job_description.txt -o output/tailored_cv.docx
```

**Note**: When job description is provided, the system will:
- Extract skills from both resume and job description
- Detect missing skills using semantic similarity
- Optimize resume with GPT-4o through 3-iteration self-improvement loop
- Calculate and display ATS compatibility scores
- Target ATS score >0.75

### With Custom Output Path and Format

```bash
python main.py input/sample_profile.txt -o output/my_cv.docx -f docx
```

### Generate PDF (requires pandoc)

```bash
python main.py input/sample_profile.txt -f pdf -o output/cv.pdf
```

### Skip Job Description Explicitly

```bash
python main.py input/sample_profile.txt -j NA
```

### Process Different Input Formats

```bash
# Process PDF
python main.py input/resume.pdf -o output/cv_from_pdf.docx

# Process Word document
python main.py input/resume.docx -o output/cv_from_docx.docx

# Process text file
python main.py input/profile.txt -o output/cv_from_txt.docx
```

## Step 5: Check Output

The generated CV will be saved in the `output/` directory. Open it with Microsoft Word, LibreOffice, or any compatible document viewer.

## System Pipeline Overview

When you run the application, the system follows this pipeline:

1. **Extract Text** - Reads your input file (PDF/DOCX/TXT)
2. **Detect Job Description** - Checks if job description is in input file
3. **Initialize Models**:
   - LLM 1 (Fast Extraction): Mistral 7B (Ollama) or Flan-T5-XL
   - LLM 2 (Optimization): GPT-4o (OpenAI)
   - Embedding Model: all-MiniLM-L6-v2 for semantic matching
4. **Extract Skills** - LLM 1 extracts skills from resume and job description
5. **Semantic Matching** - Embedding model matches skills semantically
6. **Skill Gap Detection** - Identifies missing skills (similarity threshold: 0.55)
7. **Resume Optimization** - LLM 2 (GPT-4o) optimizes resume with missing skills
8. **ATS Scoring** - Calculates ATS compatibility score (target >0.75)
9. **Self-Improvement Loop** - Automatically improves through 3 iterations until target score is reached
10. **Conversational Refinement** - You can provide feedback for iterative improvements
11. **Export** - Saves final CV to DOCX/PDF/TXT

## Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Set the API key:
```bash
export OPENAI_API_KEY="your-key"
# OR
python main.py input.txt --api-key "your-key"
```

### Issue: "Ollama not available"
**Solution**: This is normal. The system will automatically use Flan-T5-XL. To use Ollama:
1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull Mistral model: `ollama run mistral`
3. Start Ollama service: `ollama serve`
4. Verify: `curl http://localhost:11434/api/tags`

### Issue: "Model download takes too long"
**Solution**: The first run downloads Flan-T5-XL (~4GB) and embedding model. Ensure stable internet connection. Subsequent runs will be faster.

### Issue: "CUDA out of memory"
**Solution**: The system will automatically fall back to CPU. For GPU usage, ensure sufficient VRAM or use CPU mode.

### Issue: "PDF conversion failed"
**Solution**: Install pandoc for PDF conversion, or use DOCX format which works without additional tools.

## Example Commands

```bash
# Quick test with sample file
python main.py input/sample_profile.txt

# Full command with all options
python main.py input/sample_profile.txt -o output/cv.docx -f docx -j job_description.txt

# Using execution.txt
cat execution.txt
python main.py input/sample_profile.txt -o output/cv_output.docx -f docx
```

## Expected Output

When you run the program, you should see:

```
================================================================================
CV Creation using LLMs - Capstone Project
================================================================================

[Step 1] Extracting text from input document...
✓ Extracted 1234 characters from input document

[Step 2] Initializing CV Generator with LLMs...
  - Loading Ollama (Gemma 3 1B)...
  - Loading Flan-T5-XL...
✓ Primary model: flan (or ollama if available)

[Step 3] Extracting structured resume data using LLM...
✓ Resume data extracted successfully
  - Name: John Doe
  - Education entries: 1
  - Experience entries: 2
  - Skills: 15

[Step 4] Parsing job description for CV tailoring...
✓ Job description parsed
  - Keywords extracted: 8

[Step 5] Generating tailored, ATS-friendly CV using LLMs...
  - Generating Personal Information...
  - Generating Professional Summary...
  ...
✓ CV sections generated successfully

[Step 6] Formatting and saving CV...
CV saved to: output/cv_John_Doe.docx

================================================================================
CV GENERATION COMPLETE!
================================================================================
Input: input/sample_profile.txt
Output: output/cv_John_Doe.docx
Format: DOCX
Models used: flan (primary), Flan-T5-XL (secondary)
CV tailored for job requirements: Yes
================================================================================
```

## Need Help?

- Check `README.md` for detailed documentation
- Review error messages for specific issues
- Ensure all dependencies are installed correctly
- Verify input file format is supported (PDF, DOCX, TXT)
