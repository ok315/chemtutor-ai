"""
Quiz engine — the orchestrator for Phase 6.

Combines the generator, selector, and evaluator into a complete
adaptive quiz experience:

    1. Generate candidate questions for a topic
    2. Select questions matching the student's current level
    3. Grade each answer as the student submits it
    4. Compute summary score and recommend next level

The engine is stateful within a session (tracks submitted answers and
their scores), but stateless across sessions (no database, no users —
session state lives only in memory).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.quiz.generator import QuestionGenerator
from src.quiz.selector import QuestionSelector, ClassifiedQuestion
from src.quiz.evaluator import AnswerEvaluator, EvaluationResult
from src.knowledge_base.retriever import Retriever
from src.config import (
    QUIZ_FINAL_COUNT,
    QUIZ_CANDIDATE_COUNT,
    QUIZ_ADVANCE_THRESHOLD,
    QUIZ_RETRY_THRESHOLD,
    QUIZ_LEVELS,
)


@dataclass
class QuizSummary:
    """End-of-quiz summary with score and adaptive recommendation."""

    topic: str
    student_level: str
    total_questions: int
    answered_count: int
    correct_count: int          # answers with score >= 70
    partial_count: int          # answers with 30 <= score < 70
    incorrect_count: int        # answers with score < 30
    average_score: float        # 0-100
    recommendation: str         # what to do next
    next_level: Optional[str]   # suggested level for next session


@dataclass
class QuizSession:
    """
    ...
    """
    topic: str
    student_level: str
    questions: List[ClassifiedQuestion]
    answers: List = field(default_factory=list)              # was Optional[str]; now also int for MCQ
    results: List[Optional[EvaluationResult]] = field(default_factory=list)
    mode: str = "free_text"
    _engine: Optional["QuizEngine"] = field(default=None, repr=False)

    def __post_init__(self):
        # Initialize answer/result slots to None for each question
        if not self.answers:
            self.answers = [None] * len(self.questions)
        if not self.results:
            self.results = [None] * len(self.questions)

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    @property
    def answered_count(self) -> int:
        """How many questions the student has answered so far."""
        return sum(1 for a in self.answers if a is not None)

    @property
    def is_complete(self) -> bool:
        """True if the student has answered every question."""
        return self.answered_count == self.total_questions

    def submit_answer(
        self,
        question_idx: int,
        student_answer: str | int | None,
    ) -> EvaluationResult:
        """
        Submit and grade an answer to a specific question.

        For free-text mode, student_answer should be a string.
        For MCQ mode, student_answer should be an int (0-3, the selected
        option index) or None for "no selection".

        Args:
            question_idx: 0-indexed position of the question.
            student_answer: text (free-text mode) or option index (MCQ mode).

        Returns:
            The EvaluationResult.
        """
        if not 0 <= question_idx < self.total_questions:
            raise IndexError(
                f"question_idx {question_idx} out of range "
                f"(total questions: {self.total_questions})"
            )
        if self._engine is None:
            raise ValueError(
                "QuizSession not bound to an engine; cannot evaluate."
            )

        question = self.questions[question_idx]

        # Branch on MCQ vs free-text mode
        if question.is_mcq:
            # Local grading — no Gemini call
            from src.quiz.evaluator import grade_mcq_answer

            # student_answer should be an int index or None
            if isinstance(student_answer, str) and student_answer.strip().isdigit():
                idx = int(student_answer.strip())
            elif isinstance(student_answer, int):
                idx = student_answer
            else:
                idx = None

            mcq = question.mcq_data
            result = grade_mcq_answer(
                selected_index=idx,
                correct_index=mcq.correct_index,
                explanation=mcq.explanation,
            )
        else:
            # Free-text grading — uses Gemini
            answer_str = student_answer if isinstance(student_answer, str) else ""
            result = self._engine.evaluator.evaluate(
                question=question.text,
                student_answer=answer_str,
            )

        # Store the answer and result
        self.answers[question_idx] = student_answer
        self.results[question_idx] = result

        return result

    def get_summary(self) -> QuizSummary:
        """
        Compute the end-of-quiz summary.

        Can be called at any time, even partway through the quiz —
        unanswered questions are simply not counted.
        """
        graded = [r for r in self.results if r is not None]

        correct_count = sum(1 for r in graded if r.is_correct)
        partial_count = sum(1 for r in graded if r.is_partial)
        incorrect_count = sum(1 for r in graded if r.is_incorrect)

        if graded:
            average_score = sum(r.score for r in graded) / len(graded)
        else:
            average_score = 0.0

        # Decide adaptive recommendation
        recommendation, next_level = compute_recommendation(
            average_score, self.student_level
        )

        return QuizSummary(
            topic=self.topic,
            student_level=self.student_level,
            total_questions=self.total_questions,
            answered_count=self.answered_count,
            correct_count=correct_count,
            partial_count=partial_count,
            incorrect_count=incorrect_count,
            average_score=average_score,
            recommendation=recommendation,
            next_level=next_level,
        )


class QuizEngine:
    """
    The top-level orchestrator for adaptive quizzes.

    Loads all components (generator, selector, evaluator) once and
    creates QuizSession objects on demand.

    Usage:
        engine = QuizEngine()
        session = engine.start_quiz("esterification", student_level="intermediate")

        for i, q in enumerate(session.questions):
            print(f"Q{i+1}: {q.text}")
            answer = input("Your answer: ")
            result = session.submit_answer(i, answer)
            print(f"Score: {result.score}/100 — {result.feedback}")

        summary = session.get_summary()
        print(f"Total: {summary.average_score:.1f}% — {summary.recommendation}")
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        generator: QuestionGenerator | None = None,
        selector: QuestionSelector | None = None,
        evaluator: AnswerEvaluator | None = None,
    ):
        """
        Initialize the engine with all required components.

        All components are optional — if not provided, defaults are created.
        Sharing a Retriever between selector and evaluator saves memory
        (avoids loading the embedding model twice).
        """

        # Shared retriever — loads embeddings once, used by evaluator
        # (selector doesn't need retriever; classifiers are self-contained)
        if retriever is None:
            print("QuizEngine: initializing shared retriever...")
            self.retriever = Retriever()
        else:
            self.retriever = retriever

        # Generator (uses Gemini, no shared state)
        self.generator = generator or QuestionGenerator()

        # Selector (uses BERT classifiers, no retriever needed)
        self.selector = selector or QuestionSelector()

        # Evaluator (uses Gemini + retriever — share ours!)
        self.evaluator = evaluator or AnswerEvaluator(retriever=self.retriever)

        print("QuizEngine ready.")

    def start_quiz(
        self,
        topic: str,
        student_level: str = "intermediate",
        candidate_count: int = QUIZ_CANDIDATE_COUNT,
        final_count: int = QUIZ_FINAL_COUNT,
        mode: str = "free_text",
    ) -> QuizSession:
        """
        Generate and select questions for a new quiz session.

        Args:
            topic: Chemistry topic, e.g. "esterification"
            student_level: One of "beginner", "intermediate", "advanced"
            candidate_count: How many questions to generate via Gemini
            final_count: How many to select for the actual quiz
            mode: "free_text" (default, Gemini-graded) or "mcq" (locally graded)

        Returns:
            A QuizSession.
        """
        if student_level not in QUIZ_LEVELS:
            raise ValueError(
                f"Invalid student level '{student_level}'. "
                f"Must be one of {QUIZ_LEVELS}."
            )
        if mode not in ("free_text", "mcq"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'free_text' or 'mcq'."
            )

        if mode == "mcq":
            # Generate MCQs and pass them through to the selector
            print(f"Generating {candidate_count} MCQ candidates for '{topic}'...")
            mcqs = self.generator.generate_mcq(topic, count=candidate_count)
            print(f"Generated {len(mcqs)} MCQs.")

            # Selector classifies based on question TEXT but carries MCQ data
            candidate_texts = [m.text for m in mcqs]
            chosen = self.selector.select(
                candidate_texts,
                student_level=student_level,
                target_count=final_count,
                mcq_data=mcqs,
            )
        else:
            # Free-text: existing path
            print(f"Generating {candidate_count} free-text candidates for '{topic}'...")
            candidates = self.generator.generate(topic, count=candidate_count)
            print(f"Generated {len(candidates)} candidates.")
            chosen = self.selector.select(
                candidates,
                student_level=student_level,
                target_count=final_count,
            )

        print(f"Selected {len(chosen)} questions.")

        # Build the session
        session = QuizSession(
            topic=topic,
            student_level=student_level,
            questions=chosen,
            _engine=self,
        )
        # Carry mode through (we'll add this attribute to QuizSession next)
        session.mode = mode
        return session


# ============================================================
# Helper for adaptive recommendation logic
# ============================================================

def compute_recommendation(
    average_score: float, current_level: str
) -> tuple[str, Optional[str]]:
    """
    Decide what to recommend based on the student's quiz performance.

    Args:
        average_score: 0-100 average across answered questions.
        current_level: The level the student took this quiz at.

    Returns:
        (recommendation_text, next_level)
        - recommendation_text: human-readable recommendation
        - next_level: suggested level for next quiz, or None if same level
    """
    # Map level progression
    level_order = list(QUIZ_LEVELS)
    try:
        current_idx = level_order.index(current_level)
    except ValueError:
        # Unknown level — don't change anything
        return ("Quiz complete.", None)

    if average_score >= QUIZ_ADVANCE_THRESHOLD:
        # Student crushed it — suggest harder level if available
        if current_idx < len(level_order) - 1:
            next_level = level_order[current_idx + 1]
            return (
                f"Excellent work! You're ready for {next_level} level.",
                next_level,
            )
        else:
            return (
                "Outstanding! You've mastered this topic at the highest level.",
                None,
            )

    elif average_score < QUIZ_RETRY_THRESHOLD:
        # Student struggled — suggest easier level + re-explanation
        if current_idx > 0:
            next_level = level_order[current_idx - 1]
            return (
                f"This topic was challenging. We recommend reviewing the "
                f"explanation again, then trying {next_level} questions.",
                next_level,
            )
        else:
            return (
                "This topic was challenging. We recommend reviewing the "
                "explanation again before trying another quiz.",
                None,
            )

    else:
        # Mid-range — stay at the same level
        return (
            f"Good effort! Practice more {current_level} questions to "
            f"strengthen your understanding.",
            None,
        )