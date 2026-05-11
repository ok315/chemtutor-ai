"""
Retriever module.

Provides semantic search over the embedded knowledge base.
Loads chunks and embeddings from disk once, then serves
fast similarity searches for any number of queries.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np

from src.knowledge_base.embedder import embed_texts
from src.knowledge_base.kb_fetch import ensure_embeddings_file


# Default paths come from the central config
from src.config import CHUNKS_PATH as DEFAULT_CHUNKS_PATH
from src.config import EMBEDDINGS_PATH as DEFAULT_EMBEDDINGS_PATH

class Retriever:
    """
    A semantic search engine over the chemistry textbook knowledge base.

    Usage:
        retriever = Retriever()                       # loads from disk once
        results = retriever.search("electrochemistry") # returns top chunks
        for chunk in results:
            print(chunk["text"], chunk["score"])
    """

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    ):
        """
        Load the saved chunks and embeddings into memory.

        Args:
            chunks_path: Path to the chunks.json file
            embeddings_path: Path to the embeddings.npy file

        Raises:
            FileNotFoundError: If either file doesn't exist (i.e., the
                knowledge base hasn't been built yet)
            ValueError: If the number of chunks doesn't match the
                number of embedding rows
        """

        # Check files exist before trying to load them
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}\n"
                f"Run scripts/build_knowledge_base.py first."
            )
        ensure_embeddings_file(embeddings_path)
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                f"Place embeddings.npy in data/kb/ or configure CHEMTUTOR_KB_DATASET."
            )

        # Load chunks (a list of dicts with text + metadata)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)

        # Load embeddings (a 2D numpy array, already L2-normalized)
        self.embeddings: np.ndarray = np.load(embeddings_path)

        # Sanity check: every chunk must have a corresponding embedding
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.chunks)} chunks but "
                f"{self.embeddings.shape[0]} embedding rows. "
                f"Rebuild the knowledge base."
            )

        print(
            f"Retriever ready: {len(self.chunks)} chunks loaded "
            f"(embedding dimension: {self.embeddings.shape[1]})"
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Find the top_k chunks most semantically similar to the query.

        Args:
            query: The student's question or topic, e.g. "What is electrolysis?"
            top_k: How many chunks to return (default 10)

        Returns:
            A list of dicts (each chunk + its similarity score), sorted
            from most to least similar. Each dict looks like:
                {
                    "chunk_id": 42,
                    "text": "Electrolysis is the process of...",
                    "page": 87,
                    "word_count": 198,
                    "score": 0.834,    # added by this method
                }
        """

        # Step 1: Embed the query using "retrieval_query" task_type.
        # We pass [query] (a list with one item) and get back shape (1, 768).
        query_vector = embed_texts([query], task_type="retrieval_query")

        # Step 2: Compute similarity between the query and every chunk.
        # Both query and chunks are L2-normalized, so dot product = cosine similarity.
        # self.embeddings has shape (N, 768), query_vector[0] has shape (768,)
        # The result is shape (N,) — one similarity score per chunk.
        scores = self.embeddings @ query_vector[0]

        # Step 3: Find the top_k indices.
        # np.argsort sorts ascending by default; negating the array flips
        # to descending order, then we slice the first top_k.
        top_indices = np.argsort(-scores)[:top_k]

        # Step 4: Build the result list with chunk + score
        results = []
        for idx in top_indices:
            chunk_copy = dict(self.chunks[idx])  # copy to avoid mutating original
            chunk_copy["score"] = float(scores[idx])
            results.append(chunk_copy)

        return results