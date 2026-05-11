"""
End-to-end quiz engine test.

Runs a complete quiz cycle:
  1. Start a quiz on a topic
  2. Submit predefined answers
  3. Print summary with adaptive recommendation

Usage:
    python -m scripts.test_quiz_engine
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quiz.engine import QuizEngine


# Mix of answer quality so we exercise the grading + recommendation logic
TEST_TOPIC = "esterification"
TEST_LEVEL = "intermediate"

# We'll use the same answer for every question so we can predict scoring patterns.
# A mediocre answer for a chemistry quiz will score roughly mid-range.
DEFAULT_ANSWER = (
    "Esterification is a reaction between an alcohol and a carboxylic acid "
    "to form an ester and water, usually catalyzed by sulfuric acid."
)


def main():
    print("=" * 70)
    print("End-to-End Quiz Engine Test")
    print("=" * 70)

    engine = QuizEngine()

    print(f"\nStarting quiz: topic='{TEST_TOPIC}', level='{TEST_LEVEL}'")
    session = engine.start_quiz(TEST_TOPIC, student_level=TEST_LEVEL)

    print(f"\nQuiz has {session.total_questions} questions:")
    for i, q in enumerate(session.questions, start=1):
        print(f"  Q{i} [{q.difficulty}/{q.bloom}]: {q.text[:80]}...")

    print(f"\n{'='*70}")
    print(f"Submitting answers (using a generic chemistry answer for all)...")
    print(f"{'='*70}")

    for i in range(session.total_questions):
        result = session.submit_answer(i, DEFAULT_ANSWER)
        print(f"\nQ{i+1}: {session.questions[i].text[:60]}...")
        print(f"  Score:    {result.score}/100  ({result.verdict})")
        print(f"  Feedback: {result.feedback[:150]}{'...' if len(result.feedback) > 150 else ''}")

    print(f"\n{'='*70}")
    print(f"QUIZ SUMMARY")
    print(f"{'='*70}")

    summary = session.get_summary()

    print(f"  Topic:            {summary.topic}")
    print(f"  Student level:    {summary.student_level}")
    print(f"  Questions:        {summary.answered_count}/{summary.total_questions}")
    print(f"  Correct:          {summary.correct_count}")
    print(f"  Partial:          {summary.partial_count}")
    print(f"  Incorrect:        {summary.incorrect_count}")
    print(f"  Average score:    {summary.average_score:.1f}/100")
    print(f"  Next level:       {summary.next_level or 'same as current'}")
    print(f"  Recommendation:   {summary.recommendation}")


if __name__ == "__main__":
    main()