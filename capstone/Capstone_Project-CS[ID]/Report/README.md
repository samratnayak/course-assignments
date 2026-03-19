# Report Generation Instructions

This directory contains the project report for the CV Creation using LLMs capstone project.

## Files

- **Report.md**: Markdown source file containing the complete report (UPDATED with comprehensive content)
- **Report.docx**: Word document version (generated from Report.md)
- **Report.pdf**: PDF version (final submission format - needs to be regenerated from updated Report.md)
- **generate_report.py**: Python script to convert markdown to Word format
- **generate_pdf.py**: Python script to convert markdown directly to PDF (requires additional dependencies)

## Generating the Final Report

**Note**: The `Report.md` file has been updated with comprehensive information about:
- Overall approach and design philosophy
- Detailed model selection justification with evaluation process
- ResumeLM evaluation and why it wasn't used
- LangChain integration strategy and why it's used only for structured output parsing
- Enhanced workflow and module descriptions

### Option 1: Using Pandoc with LaTeX (Best Quality PDF)

1. Install LaTeX (if not already installed):
   ```bash
   # macOS
   brew install basictex
   
   # Linux
   sudo apt-get install texlive-latex-base texlive-latex-extra
   ```

2. Generate PDF:
   ```bash
   cd Report
   pandoc Report.md -o Report.pdf --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=11pt -V documentclass=article
   ```

### Option 2: Using Python Script (via DOCX)

1. Install python-docx:
   ```bash
   pip install python-docx
   ```

2. Generate DOCX:
   ```bash
   cd Report
   python3 generate_report.py
   ```

3. Convert DOCX to PDF:
   - **Option A**: Open `Report.docx` in Microsoft Word → File → Save As → PDF
   - **Option B**: Use LibreOffice (free):
     ```bash
     soffice --headless --convert-to pdf --outdir . Report.docx
     ```

### Option 3: Using Python with WeasyPrint (Direct PDF)

1. Install dependencies:
   ```bash
   pip install markdown weasyprint
   ```

2. Run PDF generator:
   ```bash
   cd Report
   python3 generate_pdf.py
   ```

### Option 4: Manual Conversion (Markdown Editor)

1. Open `Report.md` in a markdown editor that supports PDF export:
   - **Typora**: File → Export → PDF
   - **Mark Text**: File → Export → PDF
   - **VS Code**: Use "Markdown PDF" extension
   - **Online**: Use markdown-to-pdf.com or similar

2. Adjust formatting if needed (margins, font size, etc.)

### Option 5: Using Pandoc (HTML Intermediate)

1. Convert to HTML:
   ```bash
   cd Report
   pandoc Report.md -o Report.html -s --css=style.css
   ```

2. Open HTML in browser and print to PDF (Ctrl+P / Cmd+P → Save as PDF)

## Report Structure

The report follows the template structure:

1. **Project Title**: Automated CV Creation using Large Language Models
2. **Abstract**: Summary of project, objectives, methodology, and outcomes
3. **Introduction**: Context, background, and motivation
4. **Problem Statement**: Clear articulation of the problem
5. **Objectives**: Bullet point list of main goals
6. **Methodology**: 
   - Tools and technologies
   - Workflow/conceptual framework
   - Modules and functionality
   - Model selection justification
7. **Results and Analysis**: 
   - System performance metrics
   - Key findings (table format)
   - Pipeline effectiveness
8. **Conclusion**: 
   - Summary of contributions
   - Possible extensions

## Important Notes

- **Page Limit**: Maximum 3 pages (as per instructions)
- **Font Size**: Minimum 12pt
- **Format**: PDF for final submission
- **Content**: Report reflects the actual implementation with:
  - 3 LLMs (Mistral 7B, Flan-T5-XL, GPT-4o) + 1 Embedding Model
  - 15 Python modules
  - Complete pipeline architecture
  - Research-backed model justifications
  - Performance metrics and results

## Verification Checklist

Before submission, verify:

- [ ] Report is 3 pages or less
- [ ] Font size is 12pt minimum
- [ ] All sections are included
- [ ] Tables are properly formatted
- [ ] Model justifications are research-backed
- [ ] Results section includes metrics
- [ ] File is saved as PDF
- [ ] PDF is readable and properly formatted
