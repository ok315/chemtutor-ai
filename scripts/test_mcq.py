"""
Smoke test for the MCQ pipeline (no UI).

Verifies:
    1. Generator produces MCQs in the expected format
    2. Selector classifies them and carries MCQ data through
    3. Engine runs an MCQ quiz cycle end-to-end
    4. Local MCQ grading works without calling Gemini

Usage:
    python -m scripts.test_mcq_backend
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quiz.engine import QuizEngine


def main():
    print("=" * 70)
    print("MCQ Backend Test")
    print("=" * 70)

    engine = QuizEngine()

    print("\nStarting MCQ quiz...")
    session = engine.start_quiz(
        topic="esterification",
        student_level="intermediate",
        final_count=3,
        mode="mcq",
    )

    print(f"\nGot {len(session.questions)} MCQ questions:\n")
    for i, q in enumerate(session.questions, start=1):
        print(f"Q{i} [{q.difficulty}/{q.bloom}]: {q.text}")
        if q.is_mcq:
            for j, opt in enumerate(q.mcq_data.options):
                marker = "←" if j == q.mcq_data.correct_index else "  "
                letter = chr(ord("A") + j)
                print(f"  {letter}. {opt} {marker}")
            print(f"  Correct: {chr(ord('A') + q.mcq_data.correct_index)}")
            print(f"  Explanation: {q.mcq_data.explanation}")
        print()

    # Simulate student answering — pick correct for Q1, wrong for Q2/Q3
    print("=" * 70)
    print("Simulating student answers (correct, wrong, wrong)")
    print("=" * 70)

    answers_to_submit = []
    for i, q in enumerate(session.questions):
        if not q.is_mcq:
            answers_to_submit.append(0)
            continue
        # Q1: pick correct. Q2/Q3: pick first wrong option
        if i == 0:
            answers_to_submit.append(q.mcq_data.correct_index)
        else:
            wrong_idx = (q.mcq_data.correct_index + 1) % 4
            answers_to_submit.append(wrong_idx)

    for i, ans in enumerate(answers_to_submit):
        result = session.submit_answer(i, ans)
        letter = chr(ord("A") + ans)
        print(f"\nQ{i+1}: selected {letter}")
        print(f"  Score: {result.score}/100  ({result.verdict})")
        print(f"  Feedback: {result.feedback}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    summary = session.get_summary()
    print(f"  Average: {summary.average_score:.1f}/100")
    print(f"  Correct: {summary.correct_count}, "
          f"Partial: {summary.partial_count}, "
          f"Incorrect: {summary.incorrect_count}")
    print(f"  Recommendation: {summary.recommendation}")


if __name__ == "__main__":
    main()