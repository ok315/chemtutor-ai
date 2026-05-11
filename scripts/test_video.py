"""
Smoke test for the YouTube video search.

Searches for a few chemistry topics and prints the top videos
returned, so we can verify the integration is working.

Usage (from project root):
    python -m scripts.test_video
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multimodal.video import search_videos


TEST_TOPICS = [
    "esterification",
    "alkali metals",
]


def main():
    for topic in TEST_TOPICS:
        print("\n" + "=" * 70)
        print(f"TOPIC: {topic}")
        print("=" * 70)

        try:
            results = search_videos(topic)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not results:
            print("  No videos returned.")
            continue

        for i, video in enumerate(results, start=1):
            print(f"\n  [{i}] {video['title']}")
            print(f"      Channel:  {video['channel']}")
            print(f"      Duration: {video['duration']}")
            print(f"      Views:    {video['view_count']}")
            print(f"      URL:      {video['url']}")

        print()


if __name__ == "__main__":
    main()