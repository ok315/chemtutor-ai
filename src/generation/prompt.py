"""
Prompt templates for the RAG explainer.

This file holds the carefully-engineered prompt that turns retrieved
textbook chunks into a structured, grounded chemistry explanation.

Why a separate file? Prompts are not "logic" — they're more like
configuration. Putting them here means you can tune the prompt
(rephrase, add rules, restructure output) without touching any
Python logic in explainer.py.
"""

from typing import List, Dict


# ============================================================
# THE EXPLAINER PROMPT TEMPLATE
# ============================================================

# This is a "system prompt" + "user prompt" structure, all rolled
# into one big string. The LLM reads it top-to-bottom.

EXPLAINER_PROMPT_TEMPLATE = """You are ChemTutor, an expert chemistry tutor specifically designed for F.Sc (Class 12) students in Pakistan.

Your job is to explain chemistry topics from the F.Sc Chemistry textbook in a way that students can clearly understand. You must follow these strict rules:

CRITICAL RULES:
1. Use ONLY the information from the textbook passages provided below. Do NOT add information from your general knowledge that isn't in the passages.
2. If the passages don't fully answer the question, honestly say "The textbook does not cover this in detail" rather than inventing information.
3. Always cite which page(s) of the textbook your information comes from.
4. Use simple language appropriate for a Class 12 student. Avoid overly technical jargon unless the textbook uses it.
5. Use proper chemistry notation when needed (e.g., H₂SO₄, NaOH, equations).

OUTPUT FORMAT:
Structure your answer using these sections:

**Definition**
A clear, one-paragraph definition of the topic.

**Step-by-step Explanation**
A numbered list breaking down how it works or what's involved.

**Real-world Example**
A practical example mentioned in the textbook (or a logical application based on what's in the textbook).

**Key Points to Remember**
3-4 bullet points highlighting the most important takeaways for exams.

**From your textbook**
Cite the specific chapters where this information was found. Use the exact chapter labels shown in the passage headers above (e.g., "Chapter 13: Carboxylic Acids"), NOT the passage numbers.
---

TEXTBOOK PASSAGES:

{context}

---

STUDENT'S QUESTION:
{question}

---

After your explanation, add a final line in this EXACT format with no other text:
CHAPTERS_USED: <semicolon-separated list of chapter labels you actually drew information from, e.g. "CHAPTERS_USED: Chapter 2: s-Block Elements; Chapter 13: Carboxylic Acids">"""

# ============================================================
# HELPER FUNCTION TO BUILD THE FINAL PROMPT
# ============================================================

def build_explainer_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Fill in the prompt template with the student's question and
    the retrieved chunks.

    Args:
        question: What the student asked, e.g. "What is electrolysis?"
        chunks: Output from Retriever.search() — a list of chunk dicts
                with keys: text, page, score, chunk_id, etc.

    Returns:
        A complete prompt string ready to send to the LLM.

    Example output structure:
        You are ChemTutor...
        ...rules...
        TEXTBOOK PASSAGES:
        --- PASSAGE 1 (Page 49, relevance: 0.7234) ---
        Electrolysis is the process...
        --- PASSAGE 2 (Page 52, relevance: 0.6891) ---
        ...
        STUDENT'S QUESTION:
        What is electrolysis?
        Now provide your explanation...
    """

    # Format the chunks into clearly-marked passages
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        # Use chapter label for citations instead of raw PDF page number,
        # because the PDF's sequential page numbering doesn't match the
        # textbook's printed page numbers.
        chapter = chunk.get("chapter_label", "Unknown chapter")
        header = f"--- PASSAGE {i} ({chapter}, relevance: {chunk['score']:.4f}) ---"
        body = chunk["text"]
        context_blocks.append(f"{header}\n{body}")

    # Join all passages with blank lines between them for readability
    context = "\n\n".join(context_blocks)

    # Fill the template's {context} and {question} placeholders
    return EXPLAINER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )