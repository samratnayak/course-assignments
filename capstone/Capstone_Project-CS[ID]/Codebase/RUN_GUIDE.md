# How to Run the CV Creation Project

## Quick Start Guide

### Step 1: Install Dependencies

```bash
cd Codebase
pip install -r requirements.txt
```

**Note**: This will install:
- PyTorch and Transformers (for LLM 1 - local models)
- OpenAI library (for LLM 2 - GPT-4o)
- SentenceTransformers (for semantic similarity)
- Document processing libraries
- Other required packages

**First run may take time** as models will be downloaded automatically.

### Step 2: Setup OpenAI API Key (Required for LLM 2 - GPT-4o)

The project uses GPT-4o for resume optimization. You need an OpenAI API key.

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Command Line Argument**
```bash
python main.py input/sample_profile.txt --api-key "your-api-key-here"
```

**Note**: If you don't provide an API key, the system will fall back to LLM 1 (local models) for optimization, but results will be less optimal.

### Step 3: (Optional) Setup Ollama for LLM 1

For faster and better extraction, you can use Ollama with Mistral 7B:

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
- If Ollama is not available, the system will automatically use Flan-T5-XL
- The model will be downloaded on first run (may take a few minutes)

### Step 4: Run the Application

#### Basic Usage (No Job Description)
```bash
python main.py input/sample_profile.txt
```

#### With Job Description File
```bash
python main.py input/sample_profile.txt -j job_description.txt
```

#### Skip Job Description Explicitly
```bash
python main.py input/sample_profile.txt -j NA
```

#### With Custom Output
```bash
python main.py input/sample_profile.txt -o output/my_cv.docx -f docx
```

#### With OpenAI API Key
```bash
python main.py input/sample_profile.txt --api-key "sk-..." -j job_description.txt
```

#### Complete Example
```bash
python main.py input/sample_profile.txt \
  --api-key "your-openai-api-key" \
  -j job_description.txt \
  -o output/optimized_cv.docx \
  -f docx
```

## What Happens When You Run

The system follows this pipeline:

1. **Extract Text** - Reads your input file (PDF/DOCX/TXT)
2. **Detect Job Description** - Checks if job description is in input file
3. **Initialize Models**:
   - LLM 1 (Fast): Gemma 2 1B or Flan-T5-XL for extraction
   - LLM 2 (Optimization): GPT-4o for resume optimization
   - Embedding Model: all-MiniLM-L6-v2 for semantic matching
4. **Extract Skills** - LLM 1 extracts skills from resume and job description
5. **Semantic Matching** - Embedding model matches skills semantically
6. **Skill Gap Detection** - Identifies missing skills
7. **Resume Optimization** - LLM 2 (GPT-4o) optimizes resume with missing skills
8. **ATS Scoring** - Calculates ATS compatibility score
9. **Self-Improvement Loop** - Automatically improves until target score (0.8) is reached
10. **Conversational Refinement** - You can provide feedback for iterative improvements
11. **Export** - Saves final CV to DOCX/PDF/TXT

## Command Line Arguments

- `input_file` (required): Path to input file (PDF, DOCX, or TXT)
- `-o, --output`: Output file path (default: `output/cv_[name].[format]`)
- `-f, --format`: Output format - `docx`, `txt`, or `pdf` (default: `docx`)
- `-j, --job-description`: Path to job description file, or `NA` to skip
- `--api-key`: OpenAI API key (or set `OPENAI_API_KEY` environment variable)

## Example Workflow

### Scenario 1: Basic CV Generation (No Job Description)
```bash
python main.py input/sample_profile.txt
```
- System will ask if you want to provide a job description
- Type `NA` or press Enter to skip
- Generates a general CV

### Scenario 2: ATS-Optimized CV with Job Description
```bash
# Create a job description file first
echo "Software Engineer Position
Required Skills: Python, Machine Learning, Docker
..." > job_description.txt

# Run with job description
python main.py input/sample_profile.txt -j job_description.txt --api-key "sk-..."
```
- System will:
  - Extract skills from both resume and job description
  - Detect missing skills using semantic similarity
  - Optimize resume with GPT-4o
  - Run self-improvement loop
  - Show ATS scores

### Scenario 3: Job Description in Input File
If your input file contains both profile and job description (separated by "JOB DESCRIPTION" marker):
```bash
python main.py combined_input.txt
```
- System automatically detects and separates them

## Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Set the API key:
```bash
export OPENAI_API_KEY="your-key"
# OR
python main.py input.txt --api-key "your-key"
```

### Issue: "Ollama not available"
**Solution**: This is normal. System will use Flan-T5-XL. To use Ollama:
1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull Mistral model: `ollama run mistral`
3. Start Ollama service: `ollama serve`
4. Verify: `curl http://localhost:11434/api/tags`

### Issue: "Model download takes too long"
**Solution**: First run downloads Flan-T5-XL (~4GB) and embedding model. Ensure stable internet connection.

### Issue: "CUDA out of memory"
**Solution**: System automatically uses CPU. For GPU, ensure sufficient VRAM.

### Issue: "PDF conversion failed"
**Solution**: Install pandoc for PDF conversion, or use DOCX format.

## Expected Output

When running successfully, you'll see:

```
================================================================================
CV Creation using LLMs - Capstone Project
================================================================================

[Step 1] Extracting text from input document...
✓ Extracted 1234 characters from input document

[Step 2] Checking for job description in input file...
✓ No job description found in input file

[Step 3] Initializing LLM Models...
  - LLM 1 (Fast Extraction): Ollama (Gemma 2 1B) or Flan-T5-XL
  - LLM 2 (Optimization): GPT-4o (OpenAI) - Best cost-performance for resume optimization

[LLM 1 - Fast Extraction Model]
✓ LLM 1 (Extraction Model): flan

[LLM 2 - Optimization Model]
Loading OpenAI GPT-4o (gpt-4o)...
✓ OpenAI GPT-4o (gpt-4o) loaded successfully!

[Step 4] LLM 1 → Extracting Skills & Requirements from Resume...
✓ Resume data extracted successfully
  - Skills: 15
  - Tools: 8
  - All Skills: 23

[Step 5] LLM 1 → Extracting Skills & Requirements from Job Description...
✓ Job description parsed
  - JD Skills extracted: 12

[Step 6] Initializing Embedding Model for Semantic Matching...
✓ Embedding model loaded successfully!

[Step 7] Skill Gap Detection using Semantic Similarity...
✓ Skill gap analysis complete
  - Missing skills detected: 3

[Step 8] Initializing ATS Scorer...
✓ ATS Scorer initialized

[Step 9] LLM 2 → Resume Optimization with Self-Improvement Loop...
[Iteration 1]
Current ATS Score: 0.650
  - Semantic Similarity: 0.720
  - Skill Match: 0.580
  - Keyword Presence: 0.650
✓ Resume optimized

[Iteration 2]
Current ATS Score: 0.780
✓ Target score (0.8) achieved!

[Step 11] Conversational CV Refinement Mode...
[CV Preview shown]
[User feedback loop]
...

[Step 12] Formatting and saving final CV...
CV saved to: output/cv_John_Doe.docx

================================================================================
CV GENERATION COMPLETE!
================================================================================
```

## Need Help?

- Check `README.md` for detailed documentation
- Review error messages for specific issues
- Ensure all dependencies are installed
- Verify OpenAI API key is set correctly
- Check input file format is supported (PDF, DOCX, TXT)
