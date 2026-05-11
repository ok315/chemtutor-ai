"""
Embedder module.

Converts text strings into 768-dimensional vectors using a LOCAL
sentence-transformers model. Runs on CPU, no API needed, no rate limits.

Model: all-mpnet-base-v2 (420 MB, downloaded once, cached forever).
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32   # how many texts to encode per batch (CPU-friendly)


# Lazy-load the model: only download/load it the first time it's needed
# This module-level variable starts as None and gets filled on first call
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Load the model the first time it's needed, then reuse for every call.

    The first call triggers a ~420 MB download (only once ever).
    Subsequent calls return the already-loaded model instantly.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        print(f"(First-time download: ~420 MB. Cached after that.)")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded successfully.")
    return _model


def embed_texts(
    texts: List[str],
    task_type: str = "retrieval_document",
) -> np.ndarray:
    """
    Convert a list of text strings into a numpy array of embeddings.

    Args:
        texts: List of strings to embed
        task_type: Kept for API compatibility with the old Gemini-based
                   embedder. Local sentence-transformers don't need it,
                   but accepting it means the rest of our code is unchanged.

    Returns:
        A numpy array of shape (N, 768), L2-normalized.
    """

    # Validate task_type for consistency, even though we don't use it
    valid_task_types = {"retrieval_document", "retrieval_query"}
    if task_type not in valid_task_types:
        raise ValueError(
            f"task_type must be one of {valid_task_types}, got '{task_type}'"
        )

    # Get the model (loads on first call, instant after that)
    model = _get_model()

    # Encode all texts. The library handles batching internally.
    # normalize_embeddings=True does L2 normalization for us — no need
    # for our own normalize step.
    # show_progress_bar gives us a tqdm-style bar automatically.
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Cast to float32 for consistency (the library returns float32 by default,
    # but explicit is better than implicit)
    return embeddings.astype(np.float32)