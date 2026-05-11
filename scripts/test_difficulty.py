"""
Smoke test for the difficulty classifier.

Loads the trained model and runs predictions on sample chemistry questions.

Usage (from project root):
    python -m scripts.test_difficulty
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.difficulty import DifficultyClassifier


# Same questions we used in Colab — verify we get similar predictions locally
TEST_QUESTIONS = [
    "What is the symbol for sodium?",
    "What is electrolysis?",
    "Compare the reactivity of alkali metals with halogens.",
    "Predict the products of the reaction between ethanoic acid and methanol "
    "in the presence of concentrated sulfuric acid, and explain the mechanism "
    "step by step.",
    "Define molarity.",
    "Calculate the pH of a 0.01 M HCl solution.",
    "A student dissolves 5.85 g of NaCl in 250 mL of water. "
    "Calculate the molarity of the solution.",
    "Explain why noble gases are chemically inert.",
]


def main():
    print("=" * 80)
    print("Difficulty Classifier Smoke Test")
    print("=" * 80)

    # Initialize the classifier (lazy-loads the model on first predict)
    classifier = DifficultyClassifier()

    # Use batch prediction — much faster than calling predict() in a loop
    results = classifier.predict_batch(TEST_QUESTIONS)

    print()
    for question, result in zip(TEST_QUESTIONS, results):
        label = result["label"].upper()
        confidence = result["confidence"]
        probs_str = "  ".join(
            f"{lbl}={p:.2f}" for lbl, p in result["all_probs"].items()
        )

        print(f"\n[{label}] (confidence: {confidence:.2f})")
        print(f"  Q: {question}")
        print(f"  All: {probs_str}")

    print("\n" + "=" * 80)
    print(f"Classified {len(results)} questions successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()