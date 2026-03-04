# Quick Start Guide - CV Creation using LLMs

## Step 1: Install Dependencies

Navigate to the Codebase directory and install required packages:

```bash
cd "Capstone_Project-CS[ID]/Codebase"
pip install -r requirements.txt
```

**Note**: This will install PyTorch, Transformers, and other required libraries. The first run may take some time as models are downloaded.

## Step 2: (Optional) Setup Ollama for Gemma Model

If you want to use the Gemma 3 1B model via Ollama:

1. **Install Ollama**: Visit https://ollama.ai/ and install Ollama
2. **Pull the model**:
   ```bash
   ollama pull gemma2:1b
   ```
3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

**Note**: If Ollama is not available, the system will automatically use Flan-T5-XL as the primary model.

## Step 3: Run the Application

### Basic Usage

```bash
python main.py input/sample_profile.txt
```

This will:
- Extract text from the input file
- Load LLM models (Ollama if available, otherwise Flan-T5-XL)
- Extract structured resume data
- Generate CV sections
- Save output to `output/cv_[name].docx`

### With Custom Output Path

```bash
python main.py input/sample_profile.txt -o output/my_cv.docx
```

### Generate PDF (requires pandoc)

```bash
python main.py input/sample_profile.txt -f pdf -o output/cv.pdf
```

### With Job Description for Tailored CV

```bash
python main.py input/sample_profile.txt -j job_description.txt -o output/tailored_cv.docx
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

## Step 4: Check Output

The generated CV will be saved in the `output/` directory. Open it with Microsoft Word, LibreOffice, or any compatible document viewer.

## Troubleshooting

### Issue: "Ollama not available"
**Solution**: This is normal. The system will automatically use Flan-T5-XL. To use Ollama:
1. Install Ollama from https://ollama.ai/
2. Run `ollama pull gemma2:1b`
3. Ensure Ollama service is running

### Issue: "Model download takes too long"
**Solution**: The first run downloads Flan-T5-XL (~4GB). Ensure stable internet connection. Subsequent runs will be faster.

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
