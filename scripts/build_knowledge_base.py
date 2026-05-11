"""
Build script — runs the entire Phase 1 pipeline end-to-end.

Run this once after placing chemistry_book.pdf in the data/ folder.
Output: data/kb/chunks.json and data/kb/embeddings.npy

Usage (from project root):
    python -m scripts.build_knowledge_base
"""

import json
import sys
import numpy as np

# Add project root to Python path so imports work when running this script
from pathlib import Path
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
from src.knowledge_base.embedder import embed_texts


def main():
    """Run all four stages of knowledge base construction."""

    # --- Sanity check: PDF must exist ---
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Please download chemistry_book.pdf and place it in the data/ folder.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Building ChemTutor AI Knowledge Base")
    print(f"{'=' * 60}\n")

    # --- Stage 1: Parse the PDF into pages ---
    print(f"[1/4] Parsing PDF: {PDF_PATH.name}")
    pages = parse_pdf(PDF_PATH)
    print(f"      {len(pages)} non-empty pages extracted\n")

    # --- Stage 2: Chunk pages into overlapping word-windows ---
    print(f"[2/4] Chunking text (size={CHUNK_SIZE_WORDS}, overlap={CHUNK_OVERLAP_WORDS})")
    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS,
    )
    print(f"      {len(chunks)} chunks created\n")

    # --- Stage 3: Embed each chunk with Gemini ---
    print(f"[3/4] Embedding {len(chunks)} chunks via Gemini")
    print(f"      (This may take 1-3 minutes depending on book size)")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(chunk_texts, task_type="retrieval_document")
    print(f"      Embeddings shape: {embeddings.shape}\n")

    # --- Stage 4: Save chunks and embeddings to disk ---
    print(f"[4/4] Saving to disk")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"      Chunks     -> {CHUNKS_PATH}")
    print(f"      Embeddings -> {EMBEDDINGS_PATH}\n")

    print(f"{'=' * 60}")
    print(f"Knowledge base built successfully!")
    print(f"{'=' * 60}")
    print(f"  Pages parsed:   {len(pages)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Embedding dim:  {embeddings.shape[1]}")
    print(f"\nYou can now run searches via the Retriever class.\n")


if __name__ == "__main__":
    main()