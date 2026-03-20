"""
Document layout styles for exported CVs (DOCX/PDF only).

These affect formatting only — not LLM content generation.
Sample files (gibberish data) live under output/style_samples/ for visual reference.
"""

import os
from typing import Dict, List, Optional

DEFAULT_DOCUMENT_STYLE = "default"

# id -> human label + sample filename (under output/style_samples/)
_STYLE_DEFS: List[Dict[str, str]] = [
    {
        "id": "default",
        "label": "Default — current app styling (centered name, blue section headers)",
        "filename": "sample_default.docx",
        "hint": "Matches the CV you get on first generation.",
    },
    {
        "id": "sidebar_modern",
        "label": "Modern two-column — pale sidebar, sans-serif, navy role accents",
        "filename": "sample_sidebar_modern.docx",
        "hint": "Inspired by sidebar résumé layouts (contact/education/skills in left column).",
    },
    {
        "id": "classic_serif",
        "label": "Classic serif — centered header, rules under sections, Times-like",
        "filename": "sample_classic_serif.docx",
        "hint": "Traditional academic / LaTeX-style single column.",
    },
]


def get_style_samples_dir(codebase_dir: str) -> str:
    return os.path.join(codebase_dir, "output", "style_samples")


def get_style_menu_lines(codebase_dir: str) -> List[str]:
    """Console lines: option number, label, absolute sample path."""
    base = codebase_dir
    lines: List[str] = []
    for i, s in enumerate(_STYLE_DEFS, start=1):
        path = os.path.join(base, s["filename"])
        lines.append(f"  {i} — {s['label']}")
        lines.append(f"      Open sample: {path}")
        lines.append(f"      ({s['hint']})")
    return lines


def resolve_document_style(raw: str, current_id: str) -> Optional[str]:
    """
    Return new style id, or None to keep current.
    Accepts: empty, 1–N, or id substring (e.g. sidebar, classic, default).
    """
    s = (raw or "").strip()
    if not s:
        return None
    low = s.lower()
    if low.isdigit():
        idx = int(low)
        if 1 <= idx <= len(_STYLE_DEFS):
            chosen = _STYLE_DEFS[idx - 1]["id"]
            return None if chosen == current_id else chosen
        return None
    for row in _STYLE_DEFS:
        if low == row["id"]:
            return None if row["id"] == current_id else row["id"]
    matches = [row["id"] for row in _STYLE_DEFS if low in row["id"] or row["id"].startswith(low)]
    if len(matches) == 1:
        return None if matches[0] == current_id else matches[0]
    return None


def label_for_style(style_id: str) -> str:
    for row in _STYLE_DEFS:
        if row["id"] == style_id:
            return row["label"]
    return style_id


def all_document_styles() -> List[Dict[str, str]]:
    """Id, label, filename, hint for each layout (for tooling / sample build)."""
    return list(_STYLE_DEFS)
