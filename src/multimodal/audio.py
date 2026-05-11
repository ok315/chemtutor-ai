"""
Audio module — converts explanation text to spoken MP3 using gTTS.

Used by the multimodal layer to give students an audio version of
their explanation. Useful for auditory learners or when reading
on-screen text is inconvenient.
"""

import hashlib
import re
from pathlib import Path
from gtts import gTTS

from src.config import AUDIO_DIR, TTS_LANGUAGE


def clean_text_for_tts(text: str) -> str:
    """
    Strip markdown formatting and other characters that gTTS would
    read literally as gibberish.

    Args:
        text: Raw explanation text (may contain **bold**, headers, etc.)

    Returns:
        Clean text safe to feed into a text-to-speech engine.
    """
    cleaned = text

    # Remove markdown bold (**word** or __word__) — keep the inner text
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)

    # Remove markdown italics (*word* or _word_)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", cleaned)

    # Remove markdown bullet markers at line starts (*, -, +)
    cleaned = re.sub(r"^[\*\-\+]\s+", "", cleaned, flags=re.MULTILINE)

    # Remove markdown headers (# Heading)
    cleaned = re.sub(r"^#+\s+", "", cleaned, flags=re.MULTILINE)

    # Remove markdown horizontal rules (---)
    cleaned = re.sub(r"^---+$", "", cleaned, flags=re.MULTILINE)

    # Collapse multiple blank lines/whitespace into single spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def text_to_audio(
    text: str,
    output_dir: Path = AUDIO_DIR,
    language: str = TTS_LANGUAGE,
) -> Path:
    """
    Convert text into an MP3 audio file using gTTS.

    Uses a content-based filename so identical text always produces
    the same filename — letting us cache and skip re-generation when
    the same explanation is requested again.

    Args:
        text: The text to convert to speech (markdown will be stripped)
        output_dir: Folder where the MP3 will be saved
        language: gTTS language code, e.g. "en" for English, "ur" for Urdu

    Returns:
        The Path to the saved MP3 file

    Raises:
        ValueError: If text is empty after cleaning
        RuntimeError: If gTTS fails (network issue, etc.)
    """

    # Step 1: Clean the text (remove markdown that gTTS would read literally)
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        raise ValueError("Text is empty after cleaning. Nothing to convert.")

    # Step 2: Build a deterministic filename from the content's hash.
    # SHA-256 with the language baked in: same text in same language
    # always produces the same filename. Different language = different file.
    fingerprint_input = f"{language}::{clean_text}".encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_input).hexdigest()[:16]
    output_path = output_dir / f"audio_{fingerprint}.mp3"

    # Step 3: Cache hit — if the file already exists, skip regeneration
    if output_path.exists():
        return output_path

    # Step 4: Make sure the output folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Generate the audio with gTTS
    try:
        tts = gTTS(text=clean_text, lang=language, slow=False)
        tts.save(str(output_path))
    except Exception as e:
        raise RuntimeError(f"gTTS failed to generate audio: {e}") from e

    return output_path