"""
Smoke test for the question selector.

Generates candidate questions for a topic, classifies them,
then selects 5 for each student level.

Usage:
    python -m scripts.test_quiz_selector
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quiz.generator import QuestionGenerator
from src.quiz.selector import QuestionSelector


TEST_TOPIC = "esterification"


def main():
    print("=" * 70)
    print("Quiz Selector Test")
    print("=" * 70)
    print(f"Topic: {TEST_TOPIC}\n")

    print("Generating 10 candidate questions...")
    generator = QuestionGenerator()
    candidates = generator.generate(TEST_TOPIC, count=10)
    print(f"Generated {len(candidates)} candidates.\n")

    print("Initializing classifiers (this loads BERT — takes a few seconds)...")
    selector = QuestionSelector()

    # First, classify all candidates so we can see the full landscape
    classified = selector.classify(candidates)

    print("\n" + "=" * 70)
    print("ALL CANDIDATES (with classification)")
    print("=" * 70)
    for i, q in enumerate(classified, start=1):
        print(f"\n[{i:2d}] [{q.difficulty}/{q.bloom}] "
              f"(diff={q.difficulty_confidence:.2f}, bloom={q.bloom_confidence:.2f})")
        print(f"     {q.text}")

    # Now select for each level
    for level in ["beginner", "intermediate", "advanced"]:
        print("\n" + "=" * 70)
        print(f"SELECTED for level: {level.upper()}")
        print("=" * 70)
        chosen = selector.select(candidates, student_level=level)
        for i, q in enumerate(chosen, start=1):
            print(f"\n[{i}] [{q.difficulty}/{q.bloom}]")
            print(f"    {q.text}")


if __name__ == "__main__":
    main()