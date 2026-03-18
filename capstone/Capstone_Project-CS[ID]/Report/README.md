# Report Generation Instructions

This directory contains the project report for the CV Creation using LLMs capstone project.

## Files

- **Report.md**: Markdown source file containing the complete report
- **Report.docx**: Word document version (to be generated)
- **Report.pdf**: PDF version (final submission format)
- **generate_report.py**: Python script to convert markdown to Word format

## Generating the Final Report

### Option 1: Using the Python Script (Recommended)

1. Install python-docx if not already installed:
   ```bash
   pip install python-docx
   ```

2. Run the generation script:
   ```bash
   cd Report
   python generate_report.py
   ```

3. The script will create `Report.docx` from `Report.md`

4. Open `Report.docx` in Microsoft Word and:
   - Review the formatting
   - Adjust spacing if needed to fit within 3 pages
   - Ensure minimum font size is 12pt
   - Save as PDF: File → Save As → PDF

### Option 2: Using Pandoc

If you have pandoc installed:

```bash
cd Report
pandoc Report.md -o Report.docx
```

Then open in Word, review, and export to PDF.

### Option 3: Manual Conversion

1. Open `Report.md` in a markdown editor (e.g., Typora, Mark Text, or VS Code with markdown preview)
2. Export to Word format
3. Open in Microsoft Word
4. Review and adjust formatting
5. Save as PDF

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
