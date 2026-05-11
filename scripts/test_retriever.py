"""
Quick smoke test for the retriever.

Runs a few sample chemistry queries and prints the top retrieved chunks
along with their similarity scores. Use this to verify the knowledge base
is working before moving on to Phase 2.

Usage (from project root):
    python -m scripts.test_retriever
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_base.retriever import Retriever


# A handful of test queries spanning different chapters of the textbook
TEST_QUERIES = [
    "What is electrolysis?",
    "Explain the process of esterification.",
    "What are alkali metals?",
    "How does a galvanic cell work?",
    "What is the difference between alkanes and alkenes?",
]


def main():
    """Run each test query and print the top 3 retrieved chunks."""

    # Load the retriever once (loads chunks.json + embeddings.npy from disk)
    print("Loading retriever...\n")
    retriever = Retriever()
    print()

    # Loop through each test query
    for query in TEST_QUERIES:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # Retrieve top 3 chunks for this query
        results = retriever.search(query, top_k=3)

        # Print each result with its score, page number, and a text preview
        for rank, chunk in enumerate(results, start=1):
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"\n  [{rank}] Score: {chunk['score']:.4f}  Page: {chunk['page']}")
            print(f"      {preview}...")

        print()


if __name__ == "__main__":
    main()