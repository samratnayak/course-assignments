#!/usr/bin/env python3
"""
Script to generate Report.docx from Report.md
Run this script to convert the markdown report to Word format.
"""

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import os
    
    # Read the markdown content
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(script_dir, 'Report.md')
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create Word document
    doc = Document()
    
    # Set margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
    
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(12)
    
    # Parse and add content
    lines = content.split('\n')
    in_table = False
    table = None
    table_headers = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if not in_table:
                doc.add_paragraph()
            continue
        
        if line.startswith('# '):
            # Main title
            title = line[2:].strip()
            para = doc.add_heading(title, level=0)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_after = Pt(12)
            in_table = False
        elif line.startswith('## '):
            # Section heading
            heading = line[3:].strip()
            doc.add_heading(heading, level=1)
            para = doc.paragraphs[-1]
            para.paragraph_format.space_before = Pt(12)
            para.paragraph_format.space_after = Pt(6)
            in_table = False
        elif line.startswith('### '):
            # Subsection
            subheading = line[4:].strip()
            doc.add_heading(subheading, level=2)
            para = doc.paragraphs[-1]
            para.paragraph_format.space_before = Pt(10)
            para.paragraph_format.space_after = Pt(4)
            in_table = False
        elif '|' in line and line.count('|') >= 2:
            # Table row
            cells = [c.strip() for c in line.split('|') if c.strip() and c.strip() != '---']
            if 'Metric' in line or 'Value' in line or 'Notes' in line:
                # Table header
                table = doc.add_table(rows=1, cols=len(cells))
                table.style = 'Light Grid Accent 1'
                table_headers = cells
                for j, cell_text in enumerate(cells):
                    cell = table.rows[0].cells[j]
                    cell.text = cell_text
                    cell.paragraphs[0].runs[0].font.bold = True
                    cell.paragraphs[0].runs[0].font.size = Pt(11)
                in_table = True
            elif in_table and table and len(cells) == len(table_headers):
                # Table data row
                row = table.add_row()
                for j, cell_text in enumerate(cells):
                    if j < len(row.cells):
                        row.cells[j].text = cell_text
                        row.cells[j].paragraphs[0].runs[0].font.size = Pt(11)
        elif line.startswith('- '):
            # Bullet point
            text = line[2:].strip()
            # Handle bold text in bullets
            if '**' in text:
                para = doc.add_paragraph()
                parts = text.split('**')
                for j, part in enumerate(parts):
                    if j % 2 == 0:
                        para.add_run(part)
                    else:
                        run = para.add_run(part)
                        run.bold = True
            else:
                para = doc.add_paragraph(text, style='List Bullet')
            para.paragraph_format.space_after = Pt(3)
            in_table = False
        elif line.startswith('**') and line.endswith('**'):
            # Bold paragraph
            para = doc.add_paragraph()
            run = para.add_run(line[2:-2])
            run.bold = True
            in_table = False
        else:
            # Regular paragraph
            # Handle bold text
            if '**' in line:
                para = doc.add_paragraph()
                parts = line.split('**')
                for j, part in enumerate(parts):
                    if j % 2 == 0:
                        para.add_run(part)
                    else:
                        run = para.add_run(part)
                        run.bold = True
            else:
                para = doc.add_paragraph(line)
            para.paragraph_format.space_after = Pt(6)
            in_table = False
    
    # Save document
    output_path = os.path.join(script_dir, 'Report.docx')
    doc.save(output_path)
    print(f'✓ Report.docx created successfully at: {output_path}')
    print(f'  Total pages: Approximately {len(doc.paragraphs) // 25} pages')
    
except ImportError:
    print("Error: python-docx library not found.")
    print("Please install it using: pip install python-docx")
    print("\nAlternatively, you can:")
    print("1. Open Report.md in a markdown editor")
    print("2. Export/convert it to Word format")
    print("3. Or use pandoc: pandoc Report.md -o Report.docx")
except Exception as e:
    print(f"Error creating Word document: {e}")
    import traceback
    traceback.print_exc()
