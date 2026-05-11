"""
Fetch knowledge-base artifacts that Hugging Face Spaces Git rejects (.npy).

Spaces pre-receive hooks block binary blobs like numpy files; hosting the
embedding matrix on an HF Dataset and downloading at startup avoids that.

Set CHEMTUTOR_KB_DATASET to your Dataset repo id (owner/name). Reads are
anonymous for public datasets.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_embeddings_file(
    embeddings_path: Path,
    *,
    dataset_repo: str | None = None,
) -> None:
    """
    If embeddings.npy is missing, download it from a Hugging Face Dataset.

    Args:
        embeddings_path: Full path to data/kb/embeddings.npy
        dataset_repo: HF dataset repo id (e.g. ``osamaaok/chemtutor-kb``).
                      Defaults to env CHEMTUTOR_KB_DATASET, then fallback below.
    """
    if embeddings_path.exists():
        return

    repo_id = (
        (dataset_repo or "").strip()
        or os.environ.get("CHEMTUTOR_KB_DATASET", "").strip()
        or "osamaaok/chemtutor-kb"
    )
    kb_dir = embeddings_path.parent
    kb_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"Embeddings missing at {embeddings_path} and "
            f"huggingface_hub is not installed."
        ) from e

    print(
        f"Downloading embeddings.npy from Hugging Face dataset '{repo_id}' "
        "(Spaces cannot store .npy in Space Git)..."
    )
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="embeddings.npy",
        local_dir=str(kb_dir),
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Download finished but embeddings not found at {embeddings_path}."
        )
