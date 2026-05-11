"""
Central configuration for the ChemTutor AI project.

All paths and tunable parameters live here. When you want to change
chunk size, file locations, or model names, change them in this one file.
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# The project root is the parent of the parent of this file.
# config.py is at chemtutor-ai/src/config.py
# So PROJECT_ROOT is chemtutor-ai/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
# Data directories
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"

# ============================================================
# CLASSIFIER PARAMETERS (Phase 4 — BERT difficulty classifier)
# ============================================================

# Path to the fine-tuned BERT difficulty classifier
DIFFICULTY_MODEL_PATH = MODELS_DIR / "bert_difficulty"

# Maximum question length in tokens (must match training-time setting)
DIFFICULTY_MAX_LENGTH = 128

# Batch size for batch inference (fits comfortably on CPU)
DIFFICULTY_BATCH_SIZE = 16

# ============================================================
# BLOOM'S CLASSIFIER PARAMETERS (Phase 5 — hybrid rules + BERT)
# ============================================================

BLOOM_MODEL_PATH = MODELS_DIR / "bert_bloom"
BLOOM_MAX_LENGTH = 128
BLOOM_BATCH_SIZE = 16

# When BERT's confidence is below this threshold, we treat its answer
# as low-confidence. Used by the hybrid classifier to flag uncertainty.
BLOOM_LOW_CONFIDENCE_THRESHOLD = 0.65

# ============================================================
# QUIZ ENGINE PARAMETERS (Phase 6 — adaptive quiz)
# ============================================================

# How many candidate questions Gemini generates per topic.
# We generate more than we need so the selector has variety to pick from.
QUIZ_CANDIDATE_COUNT = 10

# How many questions actually appear in a quiz (selected from candidates).
QUIZ_FINAL_COUNT = 5
# Range for the configurable question count slider
QUIZ_MIN_COUNT = 3
QUIZ_MAX_COUNT = 10

# Score thresholds for adaptive level recommendation
QUIZ_ADVANCE_THRESHOLD = 80   # >= 80% → suggest harder next time
QUIZ_RETRY_THRESHOLD = 50     # < 50% → suggest easier + re-explain

# Valid student levels
QUIZ_LEVELS = ("beginner", "intermediate", "advanced")

# Specific files
PDF_PATH = DATA_DIR / "chemistry_book.pdf"
CHUNKS_PATH = KB_DIR / "chunks.json"
EMBEDDINGS_PATH = KB_DIR / "embeddings.npy"


# ============================================================
# CHUNKING PARAMETERS (from project proposal)
# ============================================================

CHUNK_SIZE_WORDS = 200      # approximate words per chunk
CHUNK_OVERLAP_WORDS = 50    # words shared between consecutive chunks


# ============================================================
# RETRIEVAL PARAMETERS
# ============================================================

TOP_K = 10  # how many chunks to retrieve per query (Stage 2 of proposal)

# ============================================================
# GENERATION PARAMETERS (Phase 2 — RAG explainer)
# ============================================================

# Gemini model used for generating explanations
# (different from the embedding model — this one is the LLM)
GEMINI_GENERATION_MODEL = "gemini-2.5-flash-lite"

# How many chunks to feed into the LLM as context for explanation
# (smaller than TOP_K because we want focused context, not a flood)
EXPLAINER_TOP_K = 5

# Generation parameters
# temperature controls randomness: 0.0 = deterministic, 1.0 = creative
# For chemistry teaching, we want low temperature (factual, consistent)
GENERATION_TEMPERATURE = 0.3
GENERATION_MAX_TOKENS = 2048

# ============================================================
# MULTIMODAL PARAMETERS (Phase 3 — Audio + Video)
# ============================================================

# Where generated audio files go (one MP3 per explanation)
AUDIO_DIR = DATA_DIR / "audio"

# gTTS language code. "en" = English, "ur" = Urdu, "hi" = Hindi
TTS_LANGUAGE = "en"

# How many YouTube videos to suggest per topic
YOUTUBE_TOP_K = 3

# Bias YouTube search toward educational content
# We append this to every search query
YOUTUBE_QUERY_SUFFIX = "F.Sc chemistry class 12 Pakistan"

# ============================================================
# ============================================================

for directory in [DATA_DIR, KB_DIR, AUDIO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)