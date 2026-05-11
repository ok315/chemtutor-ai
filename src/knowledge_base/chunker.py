"""
Chunker module.

After parsing the PDF, we get one big block of text per page.
This module slices that text into smaller, overlapping chunks
that are the right size for an embedding model to process.
"""

from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split each page's text into overlapping word-chunks.

    Args:
        pages: List of {"page": int, "text": str} dicts (output of parse_pdf)
        chunk_size: Approximate number of words per chunk (default: 200)
        overlap: Number of words shared between consecutive chunks (default: 50)

    Returns:
        A list of chunk dicts. Each chunk looks like:
            {
                "chunk_id": 0,
                "text": "Electrolysis is the process of...",
                "page": 5,
                "word_count": 200,
            }

    Raises:
        ValueError: If overlap is greater than or equal to chunk_size
    """

    # Sanity check: overlap MUST be smaller than chunk_size, otherwise
    # the window doesn't move forward and we'd loop forever
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    # How far the window moves each step.
    # Example: chunk_size=200, overlap=50 → stride=150 → each new chunk
    # starts 150 words after the previous one
    stride = chunk_size - overlap

    chunks: List[Dict] = []
    chunk_id = 0  # A unique ID for each chunk we create

    # Process each page one at a time
    for page in pages:

        # Split the page's text into a list of individual words
        # "Hello world foo" becomes ["Hello", "world", "foo"]
        words = page["text"].split()

        # If the page is empty (no words), skip it
        if not words:
            continue

        # Slide a window of size `chunk_size` across the words list,
        # advancing by `stride` each iteration
        for start in range(0, len(words), stride):

            # Get the words for this window
            # Python list slicing safely handles going past the end —
            # if start=400 and len(words)=450, words[400:600] gives words[400:450]
            window = words[start : start + chunk_size]

            # Skip very small tail chunks (the leftover at the end of a page)
            # A chunk with only 15 words is too short to be meaningful
            if len(window) < 30:
                continue

            # Save this chunk with its metadata
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(window),
                "page": page["page"],
                "word_count": len(window),
                # Carry chapter metadata through from the page so each chunk
                # knows which chapter it belongs to
                "chapter_num": page.get("chapter_num"),
                "chapter_name": page.get("chapter_name", "Unknown"),
                "chapter_label": page.get("chapter_label", "Unknown"),
            })

            chunk_id += 1
    return chunks