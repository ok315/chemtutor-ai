"""
Smoke test for the hybrid Bloom's classifier.

Tests the same 10 chemistry questions used to gate Phase 5 in Colab.
We expect the hybrid (rule + BERT fallback) to match or exceed the
gate criteria of >=7/10 correct, with HOTS questions correctly
identified.

Usage (from project root):
    python -m scripts.test_bloom
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.bloom import BloomClassifier


# Same 10 questions used to gate Phase 5
TEST_QUESTIONS = [
    # Should be LOTS (recall/explain)
    ("What is the chemical symbol for sodium?",                          "LOTS"),
    ("What is electrolysis?",                                            "LOTS"),
    ("Define molarity.",                                                 "LOTS"),
    ("Explain why noble gases are chemically inert.",                    "LOTS"),
    ("Describe the structure of an atom.",                               "LOTS"),

    # Should be HOTS (apply/analyze/etc.)
    ("Compare the reactivity of alkali metals with halogens.",           "HOTS"),
    ("Calculate the pH of a 0.01 M HCl solution.",                       "HOTS"),
    ("Predict the products of esterification between ethanoic acid and methanol.", "HOTS"),
    ("Justify the use of catalysts in industrial processes.",            "HOTS"),
    ("Design a method to separate sodium chloride from water.",          "HOTS"),
]


def main():
    print("=" * 80)
    print("Hybrid Bloom's Classifier Test")
    print("=" * 80)

    classifier = BloomClassifier()

    # Use batch prediction for efficiency
    questions = [q for q, _ in TEST_QUESTIONS]
    expected_labels = [exp for _, exp in TEST_QUESTIONS]

    results = classifier.predict_batch(questions)

    correct = 0
    rule_count = 0
    bert_count = 0

    print()
    for (q, expected), result in zip(TEST_QUESTIONS, results):
        label = result["label"]
        confidence = result["confidence"]
        method = result["method"]
        is_correct = (label == expected)
        if is_correct:
            correct += 1
        if method == "rule":
            rule_count += 1
        else:
            bert_count += 1

        marker = "✓" if is_correct else "✗"
        method_tag = f"[{method.upper()}]"

        print(f"{marker} [{label}] {method_tag} (expected {expected}, conf: {confidence:.2f})")
        print(f"  Q: {q}")

    print("\n" + "=" * 80)
    print(f"Score: {correct}/{len(TEST_QUESTIONS)} correct")
    print(f"  Rule-based decisions: {rule_count}")
    print(f"  BERT decisions:       {bert_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()