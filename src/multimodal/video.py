"""
Video module — finds related YouTube videos for a chemistry topic.

Used by the multimodal layer to give visual learners watchable
content alongside text and audio explanations.

Note: this uses youtube-search which scrapes YouTube's HTML.
If YouTube changes their layout and this stops working, we'd
switch to the official YouTube Data API v3.
"""

from typing import Dict, List
from youtube_search import YoutubeSearch

from src.config import YOUTUBE_TOP_K, YOUTUBE_QUERY_SUFFIX


# YouTube URLs from the scraper come as relative paths like "/watch?v=abc123"
# We need to prefix them with the YouTube domain to make them clickable.
YOUTUBE_BASE_URL = "https://www.youtube.com"


def search_videos(
    topic: str,
    top_k: int = YOUTUBE_TOP_K,
    query_suffix: str = YOUTUBE_QUERY_SUFFIX,
) -> List[Dict]:
    """
    Find the top YouTube videos related to a chemistry topic.

    Args:
        topic: What the student is learning, e.g. "esterification"
        top_k: How many videos to return (default 3)
        query_suffix: Extra text appended to bias toward F.Sc content

    Returns:
        A list of video info dicts. Each dict has these keys:
            - "title": video title
            - "url": full YouTube URL
            - "channel": channel name
            - "duration": "10:23" style string, or "N/A" if missing
            - "thumbnail": URL of the video thumbnail image
            - "view_count": readable string, or "N/A" if missing

    Raises:
        ValueError: If topic is empty
        RuntimeError: If the YouTube search fails
    """

    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty.")

    # Build the actual search query (topic + bias suffix)
    query = f"{topic.strip()} {query_suffix}".strip()

    # Run the search. Pad slightly because some results may have missing data.
    try:
        results_raw = YoutubeSearch(query, max_results=top_k + 2).to_dict()
    except Exception as e:
        raise RuntimeError(f"YouTube search failed: {e}") from e

    if not results_raw:
        return []

    # Convert raw video dicts into our clean format
    # Convert raw video dicts into our clean format
    videos: List[Dict] = []
    for v in results_raw:
        # The library returns relative URLs ("/watch?v=...").
        # Build the full URL so it's clickable.
        relative_url = v.get("url_suffix", "")
        full_url = f"{YOUTUBE_BASE_URL}{relative_url}" if relative_url else ""

        # Filter 1: Skip results without a working URL
        if not full_url:
            continue

        # Filter 2: Skip YouTube Shorts (vertical short-form videos).
        # They typically have URLs like /shorts/<id> and are too brief to teach a topic.
        if "/shorts/" in full_url:
            continue

        # Filter 3: Skip very short videos even if not flagged as Shorts.
        # Anything under 1 minute is unlikely to explain a chemistry topic well.
        duration = v.get("duration") or ""
        if _is_too_short(duration):
            continue

        cleaned = {
            "title": v.get("title", "Untitled"),
            "url": full_url,
            "channel": v.get("channel", "Unknown channel"),
            "duration": duration or "N/A",
            "thumbnail": _extract_thumbnail(v),
            "view_count": _clean_view_count(v.get("views") or ""),
        }

        videos.append(cleaned)

        # Stop once we have enough
        if len(videos) >= top_k:
            break

    return videos


# ============================================================
# Helpers for safely extracting nested fields from YouTube results
# ============================================================

def _extract_thumbnail(video: Dict) -> str:
    """
    Thumbnails come as a list of URL strings, smallest to largest.
    Return the largest (last in list), or empty string if missing.
    """
    thumbnails = video.get("thumbnails") or []
    if thumbnails and isinstance(thumbnails, list):
        return thumbnails[-1]
    return ""

def _is_too_short(duration: str) -> bool:
    """
    Decide whether a duration string represents a too-short video.

    YouTube duration strings look like "0:07", "1:23", "10:45", or "1:23:45".
    Anything that's less than 1 minute (no colon-then-double-digit pattern,
    or first segment is "0:XX") is rejected as too short for a real lecture.

    Args:
        duration: Duration string from YouTube, or empty.

    Returns:
        True if the video is shorter than 1 minute, False otherwise.
    """
    if not duration:
        # Missing duration — keep the video, let the user decide
        return False

    parts = duration.split(":")
    # 1 segment means seconds only, e.g. "45" → too short
    if len(parts) == 1:
        return True
    # 2 segments = minutes:seconds. If minutes is "0", it's under a minute.
    if len(parts) == 2:
        minutes = parts[0].strip()
        return minutes == "0"
    # 3 segments = hours:minutes:seconds → definitely long enough
    return False


def _clean_view_count(views: str) -> str:
    """
    Clean up view count strings that contain non-English text.

    YouTube returns localized strings like "37,991 ملاحظات" (Arabic 'views')
    based on the server's IP location. We extract just the numeric part
    and append a clean English label.

    Args:
        views: Raw view count string from YouTube.

    Returns:
        Clean string like "37,991 views", or "N/A" if no number found.
    """
    if not views:
        return "N/A"

    # Find the leading number (may include commas, dots, K, M suffixes)
    import re
    match = re.match(r"[\d.,]+\s*[KMB]?", views.strip())
    if not match:
        return "N/A"

    return f"{match.group().strip()} views"