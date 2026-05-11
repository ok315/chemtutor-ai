"""
Quiz question generator.

Generates a list of candidate quiz questions for a given chemistry topic
using Gemini. The questions span a mix of difficulty levels and cognitive
demands so the selector has variety to choose from.

Phase 4 (difficulty) and Phase 5 (Bloom's) classifiers are NOT used here —
generation produces variety, classification happens later in selector.py.
"""

import os
import re
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

from src.config import (
    GEMINI_GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    QUIZ_CANDIDATE_COUNT,
)


# Load API key (already configured in explainer.py, but we re-load to be safe)
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# Generation prompt template.
# We ask Gemini for variety across difficulty AND cognitive demand.
# We use higher temperature than the explainer so questions are diverse.
GENERATION_PROMPT_TEMPLATE = """You are a chemistry teacher writing quiz questions for F.Sc Part 2 (Class 12) students in Pakistan.

Generate exactly {count} chemistry quiz questions about the topic: "{topic}"

Requirements:
1. Mix of difficulty levels: include some easy recall questions, some medium application questions, and some hard analytical questions.
2. Mix of cognitive demands: include some "What/Define/Explain" style questions AND some "Calculate/Compare/Predict/Justify" style questions.
3. Each question must be self-contained (no references to figures, tables, or "the passage above").
4. Each question must be 1-3 sentences long, ending with a question mark or period.
5. Questions should test genuine understanding, not memorization of trivia.

Format your response as a numbered list:
1. <question 1>
2. <question 2>
3. <question 3>
...

Do NOT include answers, explanations, or any other text. ONLY the numbered list of questions."""

from dataclasses import dataclass


@dataclass
class MCQuestion:
    """A multiple-choice question with options and the correct answer."""

    text: str
    options: list[str]            # 4 options (A, B, C, D)
    correct_index: int            # 0-3, index of the correct option
    explanation: str = ""         # optional brief explanation of why


# MCQ generation prompt.
# We use a different prompt because we need structured output with options.
MCQ_GENERATION_PROMPT_TEMPLATE = """You are a chemistry teacher writing multiple-choice quiz questions for F.Sc Part 2 (Class 12) students in Pakistan.

Generate exactly {count} chemistry MCQ questions about the topic: "{topic}"

Requirements:
1. Mix of difficulty levels: include some easy recall questions, some medium application, and some hard analytical.
2. Mix of cognitive demands: "What/Define/Explain" style AND "Calculate/Compare/Predict" style.
3. Each question must be self-contained (no references to figures or "the passage above").
4. Each question must have EXACTLY 4 options labeled A, B, C, D.
5. Exactly ONE option must be correct. The other 3 should be plausible distractors.
6. Distractors should be common student mistakes, not obviously wrong.
7. Provide a brief explanation (1 sentence) of why the correct answer is right.

Format EACH question as follows (do not deviate, use these exact markers):

QUESTION_START
Q: <the question text>
A: <option A>
B: <option B>
C: <option C>
D: <option D>
CORRECT: <single letter A, B, C, or D>
EXPLANATION: <one-sentence explanation>
QUESTION_END

Generate {count} questions in this format, one after another. Do not include numbering, headers, or anything else outside the QUESTION_START/QUESTION_END blocks."""

class QuestionGenerator:
    """
    Generates quiz questions for chemistry topics using Gemini.

    Usage:
        generator = QuestionGenerator()
        questions = generator.generate("esterification", count=10)
        for q in questions:
            print(q)
    """

    def __init__(
        self,
        model_name: str = GEMINI_GENERATION_MODEL,
        # Higher temperature than explainer — we want question variety
        temperature: float = max(GENERATION_TEMPERATURE, 0.7),
    ):
        """
        Initialize the generator.

        Args:
            model_name: Which Gemini model to use.
            temperature: Generation randomness. Higher = more diverse questions.
        """
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it in your environment or "
                "add GEMINI_API_KEY=... to a .env file."
            )

        self.model_name = model_name
        self.temperature = temperature

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                # Allow plenty of tokens for 10 questions
                max_output_tokens=2000,
            ),
        )

    def generate(
        self,
        topic: str,
        count: int = QUIZ_CANDIDATE_COUNT,
    ) -> List[str]:
        """
        Generate a list of candidate questions for a topic.

        Args:
            topic: Chemistry topic, e.g. "esterification" or "alkali metals"
            count: How many questions to generate.

        Returns:
            A list of question strings, length close to (but possibly slightly
            different from) `count`. Gemini is not strict about exact counts.

        Raises:
            ValueError: If topic is empty.
            RuntimeError: If Gemini generation fails or returns no questions.
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty.")

        prompt = GENERATION_PROMPT_TEMPLATE.format(
            count=count, topic=topic.strip()
        )

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

        if not raw_text or not raw_text.strip():
            raise RuntimeError("Gemini returned an empty response.")

        questions = parse_numbered_list(raw_text)

        if not questions:
            raise RuntimeError(
                f"Couldn't parse any questions from Gemini response: {raw_text[:200]}..."
            )

        return questions
    def generate_mcq(
        self,
        topic: str,
        count: int = QUIZ_CANDIDATE_COUNT,
    ) -> list[MCQuestion]:
        """
        Generate multiple-choice questions for a topic.

        Each question has 4 options and a marked correct answer, suitable
        for instant grading without an LLM call.

        Args:
            topic: Chemistry topic.
            count: How many MCQs to generate.

        Returns:
            List of MCQuestion objects.

        Raises:
            ValueError: If topic is empty.
            RuntimeError: If Gemini fails or returns no parseable MCQs.
        """
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty.")

        prompt = MCQ_GENERATION_PROMPT_TEMPLATE.format(
            count=count, topic=topic.strip()
        )

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini MCQ generation failed: {e}") from e

        if not raw_text or not raw_text.strip():
            raise RuntimeError("Gemini returned an empty MCQ response.")

        mcqs = parse_mcq_response(raw_text)

        if not mcqs:
            raise RuntimeError(
                f"Couldn't parse any MCQs from Gemini response: {raw_text[:300]}..."
            )

        return mcqs

# ============================================================
# Helper for parsing Gemini's numbered list response
# ============================================================

def parse_numbered_list(text: str) -> List[str]:
    """
    Parse a numbered list from Gemini's response into individual questions.

    Handles formats like:
        1. What is sodium?
        2. Calculate the pH...
        3) How does electrolysis work?
        10. Compare X and Y.

    Strips numbering, trims whitespace, and skips empty lines.

    Args:
        text: Raw response from Gemini.

    Returns:
        List of cleaned question strings.
    """
    questions: List[str] = []

    # Match lines starting with "N." or "N)" where N is one or more digits.
    # The (.+?) captures the question text up to the next numbered line or end.
    # We use re.MULTILINE so ^ matches the start of each line.
    pattern = re.compile(
        r"^\s*\d+[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )

    for match in pattern.finditer(text):
        # Get the question text and clean up whitespace/newlines within it
        question = match.group(1).strip()
        # Collapse internal whitespace (multi-line questions become single-line)
        question = re.sub(r"\s+", " ", question)

        # Skip questions that are too short to be real
        if len(question) < 10:
            continue

        questions.append(question)

    return questions
def parse_mcq_response(text: str) -> list[MCQuestion]:
    """
    Parse Gemini's MCQ response into MCQuestion objects.

    Expects blocks formatted between QUESTION_START and QUESTION_END markers
    with Q:, A:, B:, C:, D:, CORRECT:, EXPLANATION: lines.

    Returns only successfully-parsed MCQs; partial/malformed ones are skipped.
    """
    mcqs: list[MCQuestion] = []

    # Split into question blocks
    blocks = re.findall(
        r"QUESTION_START\s*(.+?)\s*QUESTION_END",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    for block in blocks:
        try:
            mcq = _parse_single_mcq_block(block)
            if mcq is not None:
                mcqs.append(mcq)
        except Exception:
            # Skip malformed blocks rather than crash the whole batch
            continue

    return mcqs


def _parse_single_mcq_block(block: str) -> MCQuestion | None:
    """Parse one Q/A/B/C/D/CORRECT/EXPLANATION block. Returns None on failure."""
    # Helper: extract content after a label, up to next label or block end
    def _extract(label: str, until_labels: list[str]) -> str | None:
        # Build pattern: LABEL:\s*(content up to next label or end)
        next_labels_pattern = "|".join(rf"\n\s*{l}:" for l in until_labels)
        if next_labels_pattern:
            pattern = rf"{label}:\s*(.+?)(?={next_labels_pattern}|\Z)"
        else:
            pattern = rf"{label}:\s*(.+?)\Z"

        match = re.search(pattern, block, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    # All possible labels in order
    labels = ["Q", "A", "B", "C", "D", "CORRECT", "EXPLANATION"]

    # Extract each field; for each label, "until" includes all later labels
    fields = {}
    for i, label in enumerate(labels):
        until = labels[i + 1:]
        fields[label] = _extract(label, until)

    # Validate required fields
    if not all(fields[k] for k in ["Q", "A", "B", "C", "D", "CORRECT"]):
        return None

    # Parse correct answer letter -> index
    correct_letter = fields["CORRECT"].strip().upper()[:1]
    if correct_letter not in "ABCD":
        return None
    correct_index = ord(correct_letter) - ord("A")

    # Clean each option (collapse whitespace)
    def clean(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    return MCQuestion(
        text=clean(fields["Q"]),
        options=[
            clean(fields["A"]),
            clean(fields["B"]),
            clean(fields["C"]),
            clean(fields["D"]),
        ],
        correct_index=correct_index,
        explanation=clean(fields["EXPLANATION"]) if fields["EXPLANATION"] else "",
    )