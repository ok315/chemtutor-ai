"""
Smoke test for the RAG explainer.

Asks Gemini a few sample chemistry questions, grounded in the
F.Sc textbook chunks. Use this to verify Phase 2 works end-to-end.

Usage (from project root):
    python -m scripts.test_explainer
"""

import sys
from pathlib import Path

# Add project root to Python path so 'src.*' imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.explainer import Explainer


# Test queries spanning different chapters of the textbook
TEST_QUESTIONS = [
    "What are alkali metals?",
    "Explain the process of esterification.",
]


def main():
    """Run each test question and print the explanation."""

    # Initialize the explainer once (loads retriever + Gemini client)
    print("\nInitializing ChemTutor explainer...")
    print("=" * 70)
    explainer = Explainer()
    print("=" * 70)

    # Loop through each test question
    for question in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        # Generate the explanation
        result = explainer.explain(question)

        # Print the explanation Gemini produced
        print("\n--- EXPLANATION ---")
        print(result["explanation"])

        # Print which textbook pages were used (citations)
        print("\n--- CHAPTERS CITED ---")
        for chapter in result["sources"]:
            print(f"  - {chapter}")

        print()


if __name__ == "__main__":
    main()