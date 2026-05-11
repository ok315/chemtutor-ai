"""
BERT-based difficulty classifier.

Loads the fine-tuned model from `models/bert_difficulty/` and predicts
whether a chemistry question is easy, medium, or hard.

This module is the bridge between Phase 4's trained model and the rest
of the project (especially Phase 6's adaptive quiz engine).
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    DIFFICULTY_MODEL_PATH,
    DIFFICULTY_MAX_LENGTH,
    DIFFICULTY_BATCH_SIZE,
)


class DifficultyClassifier:
    """
    Predicts the difficulty (easy/medium/hard) of a chemistry question
    using a fine-tuned BERT model.

    Usage:
        classifier = DifficultyClassifier()                    # load once
        result = classifier.predict("What is electrolysis?")
        print(result["label"])         # "easy"
        print(result["confidence"])    # 0.92
        print(result["all_probs"])     # {"easy": 0.92, "medium": 0.07, "hard": 0.01}

        # Batch mode for many questions at once:
        results = classifier.predict_batch([
            "What is the symbol for sodium?",
            "Calculate the pH of a 0.01 M HCl solution.",
        ])
    """

    def __init__(
        self,
        model_path: Path = DIFFICULTY_MODEL_PATH,
        max_length: int = DIFFICULTY_MAX_LENGTH,
        device: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Folder containing the saved BERT model + tokenizer.
            max_length: Max tokens per question (must match training).
            device: "cuda" or "cpu". Auto-detected if None.

        Raises:
            FileNotFoundError: If the model folder doesn't exist.
        """
        self.model_path = Path(model_path)
        self.max_length = max_length

        # Defensive check: fail clearly if the model folder is missing
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Difficulty model not found at {self.model_path}. "
                f"Make sure the trained model is in models/bert_difficulty/."
            )

        # Auto-detect device: use GPU if available, else CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Lazy loading: don't actually load anything until first use
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    def _ensure_loaded(self) -> None:
        """
        Load the tokenizer and model if not already loaded.
        Idempotent — safe to call repeatedly.
        """
        if self._tokenizer is not None and self._model is not None:
            return

        print(f"Loading difficulty classifier from {self.model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        # Move to device and set to evaluation mode
        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"Difficulty classifier ready (device: {self.device}).")

    def predict(self, question: str) -> Dict:
        """
        Classify a single question.

        Args:
            question: The text of the chemistry question.

        Returns:
            A dict with keys:
              - "label": "easy" | "medium" | "hard"
              - "confidence": float in [0, 1] — probability of the predicted class
              - "all_probs": dict mapping every label to its probability

        Raises:
            ValueError: If the question is empty or whitespace-only.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Predict using the batch path so we have one code path
        results = self.predict_batch([question])
        return results[0]

    def predict_batch(self, questions: List[str]) -> List[Dict]:
        """
        Classify multiple questions in one efficient pass.

        Args:
            questions: List of question texts.

        Returns:
            List of result dicts, in the same order as the input.
            Each result has the same shape as predict()'s output.

        Raises:
            ValueError: If `questions` is empty.
        """
        if not questions:
            raise ValueError("Questions list cannot be empty.")

        self._ensure_loaded()

        results: List[Dict] = []

        # Process in chunks of DIFFICULTY_BATCH_SIZE to control memory
        for batch_start in range(0, len(questions), DIFFICULTY_BATCH_SIZE):
            batch = questions[batch_start : batch_start + DIFFICULTY_BATCH_SIZE]

            # Tokenize the whole batch at once
            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            # Move all input tensors to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference (no gradient computation = faster, less memory)
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Convert logits to probabilities (softmax along class dimension)
            probs = torch.softmax(outputs.logits, dim=1)
            # Move back to CPU for numpy-friendly access
            probs_cpu = probs.cpu().numpy()

            # Build a result dict for each question in the batch
            for i, question_probs in enumerate(probs_cpu):
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