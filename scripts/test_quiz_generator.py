"""
Smoke test for the quiz question generator.

Usage (from project root):
    python -m scripts.test_quiz_generator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quiz.generator import QuestionGenerator


TEST_TOPICS = [
    "esterification",
    "alkali metals",
]


def main():
    print("=" * 70)
    print("Quiz Question Generator Test")
    print("=" * 70)

    generator = QuestionGenerator()

    for topic in TEST_TOPICS:
        print(f"\n{'='*70}")
        print(f"TOPIC: {topic}")
        print(f"{'='*70}")

        try:
            questions = generator.generate(topic, count=10)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"\nGenerated {len(questions)} questions:\n")
        for i, q in enumerate(questions, start=1):
            print(f"  [{i:2d}] {q}")


if __name__ == "__main__":
    main()