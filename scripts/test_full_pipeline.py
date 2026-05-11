"""
End-to-end test of the full ChemTutor pipeline.

Demonstrates the complete multimodal experience:
  Question -> Retrieval -> RAG explanation -> Audio + Video links

Run from project root:
    python -m scripts.test_full_pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.explainer import Explainer
from src.multimodal.audio import text_to_audio
from src.multimodal.video import search_videos


# Use a short topic for the demo so we don't burn through quota
DEMO_QUESTION = "Explain the process of esterification."


def main():
    print("\n" + "=" * 70)
    print("ChemTutor AI — full multimodal pipeline test")
    print("=" * 70)

    # --- 1. Initialize the explainer ---
    print("\n[1/4] Initializing explainer...")
    explainer = Explainer()

    # --- 2. Generate the explanation ---
    print(f"\n[2/4] Generating explanation for:")
    print(f"      \"{DEMO_QUESTION}\"")
    result = explainer.explain(DEMO_QUESTION)

    print("\n" + "-" * 70)
    print("EXPLANATION")
    print("-" * 70)
    print(result["explanation"])

    print("\n" + "-" * 70)
    print("CHAPTERS CITED")
    print("-" * 70)
    for chapter in result["sources"]:
        print(f"  - {chapter}")

    # --- 3. Convert the explanation to audio ---
    print(f"\n[3/4] Generating audio (MP3)...")
    try:
        audio_path = text_to_audio(result["explanation"])
        print(f"      Saved -> {audio_path}")
    except Exception as e:
        print(f"      Audio failed: {e}")

    # --- 4. Find related YouTube videos ---
    # Use the original question as the search topic
    print(f"\n[4/4] Searching YouTube for related videos...")
    try:
        videos = search_videos(DEMO_QUESTION)
        if not videos:
            print("      No videos found.")
        else:
            for i, video in enumerate(videos, start=1):
                print(f"\n  [{i}] {video['title']}")
                print(f"      Channel:  {video['channel']}")
                print(f"      Duration: {video['duration']}")
                print(f"      URL:      {video['url']}")
    except Exception as e:
        print(f"      Video search failed: {e}")

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()