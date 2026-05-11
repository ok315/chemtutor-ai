"""
Hybrid Bloom's taxonomy classifier (LOTS vs HOTS).

Strategy: rule-based verb matching first (fast and reliable on common
patterns), with a fine-tuned BERT classifier as fallback for questions
that don't fit a clear pattern.

This hybrid approach trades the cleanliness of "pure ML" for production
reliability: rules handle the bulk of well-formed questions deterministically,
BERT handles the long tail.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    BLOOM_MODEL_PATH,
    BLOOM_MAX_LENGTH,
    BLOOM_BATCH_SIZE,
    BLOOM_LOW_CONFIDENCE_THRESHOLD,
)


# ============================================================
# Rule-based labeling (same logic as the training notebook)
# ============================================================

# HOTS — Higher-Order Thinking Skills
HOTS_VERBS = {
    # Apply
    "calculate", "compute", "solve", "demonstrate", "illustrate", "apply",
    # Analyze
    "analyze", "compare", "contrast", "differentiate", "distinguish",
    "examine", "investigate", "categorize", "classify", "deconstruct",
    # Evaluate
    "evaluate", "assess", "judge", "justify", "critique", "defend",
    "appraise", "argue",
    # Create
    "design", "construct", "develop", "formulate", "propose", "compose",
    "devise", "synthesize", "invent",
}

# LOTS — Lower-Order Thinking Skills
LOTS_VERBS = {
    # Remember
    "what", "who", "when", "where", "which",
    "define", "name", "list", "identify", "label", "recall", "state",
    "match", "select",
    # Understand
    "explain", "describe", "summarize", "interpret", "discuss",
    "elaborate", "outline", "paraphrase", "why",
}


# Order matters: HOTS checked first so a question with both "compare" (HOTS)
# and "what" (LOTS) gets labeled HOTS — the higher cognitive demand wins.
_BLOOM_LEVELS_ORDERED = [
    ("HOTS", HOTS_VERBS),
    ("LOTS", LOTS_VERBS),
]


def rule_based_label(text: str) -> Optional[str]:
    """
    Try to classify a question using verb-based rules.

    Returns LOTS, HOTS, or None if no Bloom's verb appears in the question.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    for level, verb_set in _BLOOM_LEVELS_ORDERED:
        if words & verb_set:
            return level
    return None


# ============================================================
# Hybrid classifier class
# ============================================================

class BloomClassifier:
    """
    Hybrid Bloom's classifier combining rule-based matching and BERT.

    Usage:
        classifier = BloomClassifier()
        result = classifier.predict("Calculate the pH of 0.01 M HCl.")
        print(result["label"])         # "HOTS"
        print(result["confidence"])    # 1.0 for rule-based, ML-based otherwise
        print(result["method"])        # "rule" or "bert"

    Lazy-loads the BERT model on first BERT-fallback call.
    """

    def __init__(
        self,
        model_path: Path = BLOOM_MODEL_PATH,
        max_length: int = BLOOM_MAX_LENGTH,
        device: Optional[str] = None,
        low_confidence_threshold: float = BLOOM_LOW_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the hybrid classifier.

        Args:
            model_path: Folder containing the BERT model + tokenizer.
            max_length: Max tokens per question (must match training).
            device: "cuda" or "cpu". Auto-detected if None.
            low_confidence_threshold: BERT predictions below this are
                flagged as low-confidence in the result dict.

        Raises:
            FileNotFoundError: If the model folder doesn't exist.
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.low_confidence_threshold = low_confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Bloom model not found at {self.model_path}. "
                f"Make sure the trained model is in models/bert_bloom/."
            )

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Lazy loading — BERT only loads on first fallback call
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    def _ensure_bert_loaded(self) -> None:
        """Load BERT and tokenizer if not already loaded. Idempotent."""
        if self._tokenizer is not None and self._model is not None:
            return

        print(f"Loading Bloom's BERT classifier from {self.model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"Bloom's classifier ready (device: {self.device}).")

    def _predict_bert(self, questions: List[str]) -> List[Dict]:
        """
        Run BERT inference on a batch of questions.
        Returns list of dicts with label/confidence/all_probs.
        """
        self._ensure_bert_loaded()

        results: List[Dict] = []

        for batch_start in range(0, len(questions), BLOOM_BATCH_SIZE):
            batch = questions[batch_start : batch_start + BLOOM_BATCH_SIZE]

            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            probs_cpu = probs.cpu().numpy()

            for question_probs in probs_cpu:
                predicted_id = int(question_probs.argmax())
                predicted_label = self._model.config.id2label[predicted_id]

                all_probs = {
                    self._model.config.id2label[j]: float(p)
                    for j, p in enumerate(question_probs)
                }

                results.append({
                    "label": predicted_label,
                    "confidence": float(question_probs[predicted_id]),
                    "all_probs": all_probs,
                })

        return results

    def predict(self, question: str) -> Dict:
        """
        Classify a single question using the hybrid strategy.

        Args:
            question: The text of the question.

        Returns:
            A dict with keys:
              - "label": "LOTS" or "HOTS"
              - "confidence": float in [0, 1]
              - "method": "rule" or "bert"
              - "all_probs": dict mapping each label to probability
                             (for BERT predictions; rule predictions show 1.0/0.0)
              - "low_confidence": bool — True if confidence is below threshold

        Raises:
            ValueError: If question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # === Try rules first ===
        rule_label = rule_based_label(question)
        if rule_label is not None:
            other_label = "LOTS" if rule_label == "HOTS" else "HOTS"
            return {
                "label": rule_label,
                "confidence": 1.0,
                "method": "rule",
                "all_probs": {rule_label: 1.0, other_label: 0.0},
                "low_confidence": False,
            }

        # === Fall back to BERT ===
        bert_result = self._predict_bert([question])[0]
        bert_result["method"] = "bert"
        bert_result["low_confidence"] = (
            bert_result["confidence"] < self.low_confidence_threshold
        )
        return bert_result

    def predict_batch(self, questions: List[str]) -> List[Dict]:
        """
        Classify multiple questions efficiently.

        Splits questions into rule-handled and BERT-handled groups,
        processes each group with the appropriate method, then
        reassembles results in the original order.

        Args:
            questions: List of question texts.

        Returns:
            List of result dicts in the same order as the input.

        Raises:
            ValueError: If `questions` is empty.
        """
        if not questions:
            raise ValueError("Questions list cannot be empty.")

        # First pass: try rules on every question
        # We track which ones need BERT (rule returned None)
        # and remember their original positions so we can reassemble.
        results: List[Optional[Dict]] = [None] * len(questions)
        bert_indices: List[int] = []
        bert_questions: List[str] = []

        for i, question in enumerate(questions):
            rule_label = rule_based_label(question)
            if rule_label is not None:
                other = "LOTS" if rule_label == "HOTS" else "HOTS"
                results[i] = {
                    "label": rule_label,
                    "confidence": 1.0,
                    "method": "rule",
                    "all_probs": {rule_label: 1.0, other: 0.0},
                    "low_confidence": False,
                }
            else:
                bert_indices.append(i)
                bert_questions.append(question)

        # Second pass: run BERT on the unmatched questions in one batch
        if bert_questions:
            bert_results = self._predict_bert(bert_questions)
            for i, result in zip(bert_indices, bert_results):
                result["method"] = "bert"
                result["low_confidence"] = (
                    result["confidence"] < self.low_confidence_threshold
                )
                results[i] = result

        # All slots should now be filled
        return results  # type: ignore