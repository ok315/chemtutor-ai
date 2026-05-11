"""
PDF parser module.

Reads a PDF file and returns its text page-by-page.
Also detects chapter information from page headers so chunks can
later cite "Chapter 13: Carboxylic Acids" instead of arbitrary
PDF page numbers.
"""

import re
from pathlib import Path
from typing import List, Dict
import pdfplumber
from tqdm import tqdm


# Regex to detect chapter headers in this specific textbook.
# Examples this matches:
#   "2. s-Block Elements eLearn.Punjab"           -> num=2, name="s-Block Elements"
#   "5. Halogens and Noble Gases eLearn.Punjab"   -> num=5, name="Halogens and Noble Gases"
#   "13. CARBOXYLIC ACIDS eLearn.Punjab"          -> num=13, name="CARBOXYLIC ACIDS"
#   "Group VA and VIA Elements eLearn.Punjab"     -> num=None, name="Group VA and VIA Elements"
CHAPTER_HEADER_RE = re.compile(
    r"^\s*(?:(\d+)\.\s+)?([A-Za-z][A-Za-z0-9\s\-,]+?)\s+eLearn\.Punjab",
    re.IGNORECASE,
)


def parse_pdf(pdf_path: Path) -> List[Dict]:
    """
    Read every page of a PDF and extract its text + chapter metadata.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        A list where each item is a dictionary like:
            {
                "page": 1,
                "text": "Chapter content...",
                "chapter_num": 2,                 # int or None for ch.4
                "chapter_name": "s-Block Elements",
                "chapter_label": "Chapter 2: s-Block Elements",
            }
    """
    pages = []
    current_chapter_num: int | None = None
    current_chapter_name: str = "Front matter"

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="Parsing PDF")):
            raw_text = page.extract_text() or ""

            # Try to detect a chapter header in the first ~150 chars.
            # If found, update our running "current chapter" tracker.
            header_match = CHAPTER_HEADER_RE.search(raw_text[:150])
            if header_match:
                num_str, name = header_match.groups()
                # Update both num and name together. None is fine for ch.4
                # which has no number in its header.
                current_chapter_num = int(num_str) if num_str is not None else None
                current_chapter_name = name.strip().title()

            cleaned_text = clean_text(raw_text)

            if cleaned_text.strip():
                pages.append({
                    "page": i + 1,
                    "text": cleaned_text,
                    "chapter_num": current_chapter_num,
                    "chapter_name": current_chapter_name,
                    "chapter_label": format_chapter_label(
                        current_chapter_num, current_chapter_name
                    ),
                })

    return pages


def format_chapter_label(num: int | None, name: str) -> str:
    """
    Produce a human-readable chapter label.

    Examples:
        format_chapter_label(2, "s-Block Elements")  -> "Chapter 2: s-Block Elements"
        format_chapter_label(None, "Group VA and VIA Elements")
            -> "Chapter: Group VA and VIA Elements"
        format_chapter_label(None, "Front matter")   -> "Front matter"
    """
    if name == "Front matter":
        return name
    if num is None:
        return f"Chapter: {name}"
    return f"Chapter {num}: {name}"


def clean_text(text: str) -> str:
    """
    Normalize whitespace in extracted text.

    Args:
        text: Raw text as extracted by pdfplumber

    Returns:
        Cleaned text with normalized whitespace
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return " ".join(lines)