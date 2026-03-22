"""
Generate sample DOCX files (gibberish content) for each document layout style.
Run from Codebase:  python build_style_samples.py

Output: output/style_samples/sample_*.docx
"""

import os

from cv_document_styles import all_document_styles, get_style_samples_dir
from cv_formatter import CVFormatter

_GIBBERISH = {
    "Personal Information": (
        "Qwerty Z. Lorem\n"
        "qwerty.lorem@example.invalid\n"
        "+0 555 0100 9999\n"
        "linkedin.com/in/qwertylorem\n"
        "123 Fictional Ave, Nowhere"
    ),
    "Professional Summary": (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Pretend this is a 100-word professional summary full of placeholder text "
        "for layout preview only — not real candidate data."
    ),
    "Work Experience": (
        "SENIOR PLACEHOLDER ENGINEER, ACME CORP LOREM INC\n"
        "2022 – Present\n"
        "- Built imaginary systems using foobar and widgets\n"
        "- Led a team of null pointers to 99% uptime in demo environments\n\n"
        "JUNIOR GIBBERISH DEVELOPER, SAMPLE LLC\n"
        "2019 – 2021\n"
        "- Wrote lorem code and ipsum tests\n"
        "- Migrated legacy nonsense to modern nonsense"
    ),
    "Education": (
        "M.S. Nonsense Studies, University of Placeholder — 2019\n"
        "B.S. Gibberish Science, College of Example — 2017"
    ),
    "Skills": (
        "Languages: Lorem, Ipsum, Dolor\n"
        "Tools: Foobar, Widget, PlaceholderDB"
    ),
    "Certifications": "Certified Imaginary Professional (2023)",
    "Projects": (
        "Project Alpha — Built a demo using fake metrics and sample dashboards.\n"
        "Project Beta — Open-source gibberish toolkit for layout testing."
    ),
    "Languages": "English (Professional), Latin (Classical, sample only)",
}


def main() -> None:
    """Emit one sample DOCX per layout into ``output/style_samples`` for UI preview."""
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = get_style_samples_dir(base)
    os.makedirs(out_dir, exist_ok=True)
    fmt = CVFormatter()
    for row in all_document_styles():
        sid = row["id"]
        fname = row["filename"]
        path = os.path.join(out_dir, fname)
        fmt.format_to_docx(
            _GIBBERISH,
            path,
            candidate_name="Sample Q. Candidate",
            document_style=sid,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
