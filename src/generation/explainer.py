"""
The RAG Explainer.

Takes a chemistry topic from the student, retrieves relevant textbook
passages, and uses Google Gemini to generate a structured, grounded
explanation.

This is the heart of Phase 2 — it ties together retrieval (Phase 1)
and generation (LLM).
"""

import os
import re
from typing import Dict, List
import google.generativeai as genai
from dotenv import load_dotenv

from src.config import (
    GEMINI_GENERATION_MODEL,
    EXPLAINER_TOP_K,
    GENERATION_TEMPERATURE,
    GENERATION_MAX_TOKENS,
)
from src.knowledge_base.retriever import Retriever
from src.generation.prompt import build_explainer_prompt


# Load .env for local development (does not override existing OS env vars)
load_dotenv()


def _require_gemini_api_key() -> str:
    """Return Gemini API key from the environment (never hardcoded)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Export it in your environment or add "
            "GEMINI_API_KEY=... to a .env file (see README)."
        )
    return key


class Explainer:
    """
    Generates textbook-grounded explanations for chemistry topics.

    Usage:
        explainer = Explainer()                    # load once at startup
        result = explainer.explain("What is electrolysis?")
        print(result["explanation"])               # the formatted answer
        print(result["sources"])                   # which chapters were used
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        model_name: str = GEMINI_GENERATION_MODEL,
        top_k: int = EXPLAINER_TOP_K,
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
    ):
        """
        Initialize the explainer.

        Args:
            retriever: An existing Retriever instance. If None, a new one
                is created (loading chunks + embeddings from disk).
            model_name: Which Gemini model to use for generation.
            top_k: How many chunks to retrieve and feed into the prompt.
            temperature: Generation randomness (0.0 = deterministic).
            max_tokens: Maximum length of the generated response.
        """

        # Load the retriever (or use the one passed in).
        # Allowing an existing retriever to be reused saves memory if
        # multiple components share the knowledge base.
        if retriever is None:
            print("Initializing retriever for explainer...")
            self.retriever = Retriever()
        else:
            self.retriever = retriever

        # Save settings as attributes
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = _require_gemini_api_key()
        genai.configure(api_key=api_key)

        # Create the Gemini client.
        # generation_config defines the LLM's behavior for ALL future calls
        # we make through this client.
        print(f"Initializing Gemini client (model: {model_name})...")
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        print("Explainer ready.")

    def explain(self, question: str) -> Dict:
        """
        Generate a grounded explanation for a chemistry question.

        Args:
            question: The student's question, e.g. "What is electrolysis?"

        Returns:
            A dictionary with these keys:
                - "question": the original question (echoed back)
                - "explanation": the formatted explanation text (cleaned)
                - "sources": list of chapter labels Gemini actually used
                - "retrieved_chapters": all chapters present in retrieved chunks
                - "chunks_used": the actual chunks fed to the LLM (for debugging)

        Raises:
            ValueError: If question is empty or whitespace-only
            RuntimeError: If Gemini's response is empty or malformed
        """

        # Validate input
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # --- Step 1: Retrieve relevant chunks ---
        chunks = self.retriever.search(question, top_k=self.top_k)

        # --- Step 2: Build the prompt with chunks + question ---
        prompt = build_explainer_prompt(question=question, chunks=chunks)

        # --- Step 3: Send the prompt to Gemini ---
        try:
            response = self.model.generate_content(prompt)
            explanation = response.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

        # Sanity check: explanation must not be empty
        if not explanation or not explanation.strip():
            raise RuntimeError(
                "Gemini returned an empty response. "
                "This may be due to safety filters or an API issue."
            )

        # --- Step 4: Parse structured citations and package the result ---

        # Chapters Gemini said it actually used (parsed from CHAPTERS_USED: line)
        chapters_cited_by_llm = parse_chapters_used(explanation)

        # Chapters from the chunks we retrieved (deduplicated, ordered)
        retrieved_chapters: List[str] = []
        seen = set()
        for chunk in chunks:
            label = chunk.get("chapter_label", "Unknown")
            if label not in seen:
                retrieved_chapters.append(label)
                seen.add(label)

        # Validate: every chapter Gemini cited should be in our retrieved chunks.
        # If not, Gemini hallucinated a chapter name.
        hallucinated_chapters = [
            c for c in chapters_cited_by_llm if c not in retrieved_chapters
        ]
        if hallucinated_chapters:
            print(
                f"WARNING: Gemini cited chapters not in retrieved chunks: "
                f"{hallucinated_chapters}"
            )

        # Real sources = intersection of cited and retrieved.
        # Fall back to all retrieved chapters if Gemini didn't cite anything parseable.
        sources = [
            c for c in chapters_cited_by_llm if c in retrieved_chapters
        ] or retrieved_chapters

        # Strip the machine-parseable trailer from the student-facing explanation
        clean_explanation = strip_chapters_used_line(explanation)

        return {
            "question": question,
            "explanation": clean_explanation,
            "sources": sources,
            "retrieved_chapters": retrieved_chapters,
            "chunks_used": chunks,
        }


# ============================================================
# Helpers for parsing the structured citation trailer
# ============================================================

def parse_chapters_used(text: str) -> List[str]:
    """
    Extract the structured CHAPTERS_USED line from Gemini's response.

    Looks for a line like
        "CHAPTERS_USED: Chapter 2: s-Block Elements; Chapter 13: Carboxylic Acids"
    and returns the list of chapter labels.

    Args:
        text: The full text Gemini returned.

    Returns:
        List of chapter label strings, or empty list if missing/unparseable.
    """
    match = re.search(r"CHAPTERS_USED:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return []

    raw = match.group(1).strip()
    chapters: List[str] = []

    # Split on semicolons (chapter labels themselves contain commas)
    for piece in raw.split(";"):
        piece = piece.strip()
        if piece:
            chapters.append(piece)

    return chapters


def strip_chapters_used_line(text: str) -> str:
    """
    Remove the CHAPTERS_USED line from the explanation text so the
    student doesn't see the machine-parseable trailer.

    Args:
        text: The full text Gemini returned.

    Returns:
        Same text with the CHAPTERS_USED line removed.
    """
    cleaned = re.sub(
        r"\s*CHAPTERS_USED:.*?(?:\n|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()