"""
Smoke test for the answer evaluator.

Tests the same question with three different answers (good, partial, bad)
to verify the grading rubric works correctly.

Usage:
    python -m scripts.test_quiz_evaluator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quiz.evaluator import AnswerEvaluator


QUESTION = "What is electrolysis?"

TEST_ANSWERS = [
    # (description, answer)
    ("FULLY CORRECT",
     "Electrolysis is a chemical decomposition process where an electric current "
     "is passed through a liquid or solution containing ions, causing a non-spontaneous "
     "redox reaction. It splits compounds into simpler substances at the electrodes."),

    ("PARTIALLY CORRECT",
     "It is breaking down compounds using electricity."),

    ("MOSTLY WRONG",
     "Electrolysis is the same as photosynthesis — it produces oxygen from water using sunlight."),

    ("OFF-TOPIC",
     "I don't know."),
]


def main():
    print("=" * 70)
    print("Answer Evaluator Test")
    print("=" * 70)
    print(f"Question: {QUESTION}\n")

    evaluator = AnswerEvaluator()

    for description, answer in TEST_ANSWERS:
        print("=" * 70)
        print(f"TEST: {description}")
        print(f"Student answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print("-" * 70)

        try:
            result = evaluator.evaluate(QUESTION, answer)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Score:    {result.score}/100")
        print(f"  Verdict:  {result.verdict}")
        print(f"  Correct?  {result.is_correct}  (partial: {result.is_partial})")
        print(f"  Feedback: {result.feedback}")
        print()


if __name__ == "__main__":
    main()