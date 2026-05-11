"""
Refresh chunk metadata (chapter labels) without re-embedding.

The embeddings depend only on the chunk TEXT, which we're not changing.
We just need to re-parse the PDF with the updated chapter detector and
re-chunk. The embeddings.npy file stays as-is.

Usage (from project root):
    python -m scripts.refresh_chapter_metadata
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PDF_PATH,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)
from src.knowledge_base.pdf_parser import parse_pdf
from src.knowledge_base.chunker import chunk_pages


def main():
    print("\n[1/3] Re-parsing PDF with chapter detection")
    pages = parse_pdf(PDF_PATH)
    print(f"      {len(pages)} pages parsed")

    # Quick sanity check — show how many chapters we detected
    chapters = sorted({p["chapter_label"] for p in pages})
    print(f"      Detected {len(chapters)} chapter labels:")
    for ch in chapters:
        print(f"        - {ch}")

    print(f"\n[2/3] Re-chunking (size={CHUNK_SIZE_WORDS}, overlap={CHUNK_OVERLAP_WORDS})")
    chunks = chunk_pages(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    print(f"      {len(chunks)} chunks created")

    # Critical: chunk count must match existing embeddings count
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(chunks) != embeddings.shape[0]:
        print(
            f"\nERROR: Chunk count ({len(chunks)}) doesn't match embedding count "
            f"({embeddings.shape[0]}). The chunker output has changed shape."
        )
        print("You need to do a full rebuild instead:")
        print("  python -m scripts.build_knowledge_base")
        sys.exit(1)

    print(f"\n[3/3] Saving updated chunks (embeddings unchanged)")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"      Saved -> {CHUNKS_PATH}")
    print(f"\nDone. {len(chunks)} chunks now have chapter metadata.")


if __name__ == "__main__":
    main()