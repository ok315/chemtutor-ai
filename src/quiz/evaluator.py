"""
Answer evaluator.

Grades a student's answer to a chemistry question using:
  1. Retrieval (Phase 1) — fetch relevant textbook chunks for ground truth
  2. Gemini (Phase 2) — score the answer against the textbook content

Returns a 0-100 score and human-readable feedback.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

from src.config import (
    GEMINI_GENERATION_MODEL,
    EXPLAINER_TOP_K,
)
from src.knowledge_base.retriever import Retriever


# Load API key
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# Evaluation prompt template.
# Low temperature for consistent grading — same answer should get the same score.
EVALUATION_PROMPT_TEMPLATE = """You are a chemistry teacher grading a Pakistani F.Sc Class 12 student's answer. Be fair and educational.

CONTEXT FROM TEXTBOOK:
{context}

QUESTION:
{question}

STUDENT'S ANSWER:
{answer}

GRADE THIS ANSWER:
- Use a 0-100 scale where:
  - 90-100: Fully correct with good explanation
  - 70-89: Mostly correct, may have minor errors or missing details
  - 50-69: Partially correct, demonstrates some understanding
  - 30-49: Mostly wrong but shows some related knowledge
  - 0-29: Wrong or off-topic
- Be reasonable — students may use different valid phrasings than the textbook
- Reward partial credit for partially correct answers
- Don't penalize minor spelling or formatting errors

Format your response EXACTLY as follows (do not deviate):

SCORE: <number from 0 to 100>
VERDICT: <correct | partial | incorrect>
FEEDBACK: <2-3 sentences explaining what was right, what was wrong, and what the student can improve>"""


@dataclass
class EvaluationResult:
    """Result of grading a single answer."""

    score: int                  # 0-100
    verdict: str                # "correct", "partial", or "incorrect"
    feedback: str               # Human-readable explanation

    @property
    def is_correct(self) -> bool:
        """True if the verdict is 'correct'."""
        return self.verdict == "correct"

    @property
    def is_partial(self) -> bool:
        """True if the verdict is 'partial'."""
        return self.verdict == "partial"

    @property
    def is_incorrect(self) -> bool:
        """True if the verdict is 'incorrect'."""
        return self.verdict == "incorrect"


class AnswerEvaluator:
    """
    Grades student answers using textbook-grounded Gemini scoring.

    Usage:
        evaluator = AnswerEvaluator()
        result = evaluator.evaluate(
            question="What is electrolysis?",
            student_answer="Breaking down compounds using electricity",
        )
        print(result.score)      # e.g., 75
        print(result.verdict)    # "partial"
        print(result.feedback)   # "Good basic understanding..."

    Reuses an existing retriever if provided to save memory.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        model_name: str = GEMINI_GENERATION_MODEL,
        top_k: int = EXPLAINER_TOP_K,
        # Low temperature for consistent grading
        temperature: float = 0.1,
    ):
        """
        Initialize the evaluator.

        Args:
            retriever: Existing Retriever (preferred — saves loading the
                embeddings). If None, creates one.
            model_name: Which Gemini model for grading.
            top_k: How many textbook chunks to retrieve as context.
            temperature: Generation temperature. Low for consistent grades.
        """
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it in your environment or "
                "add GEMINI_API_KEY=... to a .env file."
            )

        if retriever is None:
            print("Initializing retriever for evaluator...")
            self.retriever = Retriever()
        else:
            self.retriever = retriever

        self.model_name = model_name
        self.top_k = top_k
        self.temperature = temperature

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=500,
            ),
        )

    def evaluate(self, question: str, student_answer: str) -> EvaluationResult:
        """
        Grade a single student answer.

        Args:
            question: The quiz question.
            student_answer: The student's answer text.

        Returns:
            An EvaluationResult with score, verdict, feedback.

        Raises:
            ValueError: If question is empty.
            RuntimeError: If Gemini fails or response is unparseable.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if not student_answer or not student_answer.strip():
            # Empty answer = automatic 0
            return EvaluationResult(
                score=0,
                verdict="incorrect",
                feedback="No answer provided.",
            )

        # Step 1: Retrieve textbook context for grounding the grade
        chunks = self.retriever.search(question, top_k=self.top_k)
        context_text = "\n\n".join(
            f"[{chunk.get('chapter_label', 'Unknown')}]\n{chunk['text']}"
            for chunk in chunks
        )

        # Step 2: Build the grading prompt
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question.strip(),
            answer=student_answer.strip(),
        )

        # Step 3: Send to Gemini
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini grading failed: {e}") from e

        if not raw_text or not raw_text.strip():
            raise RuntimeError("Gemini returned empty grading response.")

        # Step 4: Parse the structured response
        return parse_evaluation_response(raw_text)


# ============================================================
# Helper for parsing Gemini's structured grading response
# ============================================================

def parse_evaluation_response(text: str) -> EvaluationResult:
    """
    Parse Gemini's structured grading response into an EvaluationResult.

    Expected format:
        SCORE: <0-100>
        VERDICT: <correct | partial | incorrect>
        FEEDBACK: <text>

    If parsing fails on any field, falls back to defensive defaults rather
    than crashing.
    """
    # Extract score (look for number after "SCORE:")
    score_match = re.search(r"SCORE:\s*(\d+)", text, flags=re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        score = max(0, min(100, score))    # clamp to [0, 100]
    else:
        # Couldn't parse — fall back to neutral score
        score = 50

    # Extract verdict
    verdict_match = re.search(
        r"VERDICT:\s*(correct|partial|incorrect)",
        text,
        flags=re.IGNORECASE,
    )
    if verdict_match:
        verdict = verdict_match.group(1).lower()
    else:
        # Derive from score if missing
        if score >= 70:
            verdict = "correct"
        elif score >= 30:
            verdict = "partial"
        else:
            verdict = "incorrect"

    # Extract feedback (everything after "FEEDBACK:")
    feedback_match = re.search(
        r"FEEDBACK:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if feedback_match:
        feedback = feedback_match.group(1).strip()
        # Collapse internal whitespace for cleaner display
        feedback = re.sub(r"\s+", " ", feedback)
    else:
        # Fallback: if FEEDBACK marker is missing, take any text after the
        # last marker we found (SCORE or VERDICT). This handles cases where
        # Gemini formats the response slightly differently.
        last_marker_end = 0
        for marker_pattern in [r"VERDICT:.*?\n", r"SCORE:.*?\n"]:
            for m in re.finditer(marker_pattern, text, flags=re.IGNORECASE):
                last_marker_end = max(last_marker_end, m.end())

        if last_marker_end > 0:
            feedback = text[last_marker_end:].strip()
            feedback = re.sub(r"\s+", " ", feedback)
            # Also strip any "FEEDBACK:" prefix that might be there
            feedback = re.sub(
                r"^FEEDBACK:\s*", "", feedback, flags=re.IGNORECASE
            )
        else:
            feedback = "Unable to parse detailed feedback."

    return EvaluationResult(
        score=score,
        verdict=verdict,
        feedback=feedback,
    )

def grade_mcq_answer(
    selected_index: int | None,
    correct_index: int,
    explanation: str = "",
) -> EvaluationResult:
    """
    Grade an MCQ answer locally — no Gemini call needed.

    Args:
        selected_index: 0-3, what the student chose. None means no selection.
        correct_index: 0-3, the correct option.
        explanation: Optional explanation of why the answer is correct.

    Returns:
        An EvaluationResult with score 100 (correct) or 0 (incorrect).
    """
    if selected_index is None:
        return EvaluationResult(
            score=0,
            verdict="incorrect",
            feedback="No option selected.",
        )

    if selected_index == correct_index:
        feedback = "✅ Correct!"
        if explanation:
            feedback += f" {explanation}"
        return EvaluationResult(
            score=100,
            verdict="correct",
            feedback=feedback,
        )
    else:
        correct_letter = chr(ord("A") + correct_index)
        feedback = f"❌ Incorrect. The correct answer is {correct_letter}."
        if explanation:
            feedback += f" {explanation}"
        return EvaluationResult(
            score=0,
            verdict="incorrect",
            feedback=feedback,
        )