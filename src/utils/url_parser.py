import re
from urllib.parse import urlparse, parse_qs, unquote

# Valid YouTube domains
YOUTUBE_DOMAINS = frozenset(["youtube.com", "youtu.be", "youtube-nocookie.com"])

# Valid Bilibili domains
BILIBILI_DOMAINS = frozenset(["bilibili.com", "b23.tv"])

# Valid Apple Podcasts domains
APPLE_PODCASTS_DOMAINS = frozenset(["podcasts.apple.com"])

# Valid Xiaoyuzhou (小宇宙) domains
XIAOYUZHOU_DOMAINS = frozenset(["xiaoyuzhoufm.com"])


def _is_youtube_host(hostname: str) -> bool:
    """
    Check if a hostname is a valid YouTube domain.

    Uses strict matching to prevent bypass attacks like youtube.com.evil.com
    """
    if not hostname:
        return False

    hostname = hostname.lower()

    # Check exact match or valid subdomain
    for domain in YOUTUBE_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return True

    return False


def extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from various URL formats.

    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://youtube.com/v/VIDEO_ID
    - https://www.youtube.com/live/VIDEO_ID

    Args:
        url: The YouTube URL to parse

    Returns:
        The video ID or None if not found
    """
    if not url:
        return None

    url = url.strip()

    # Try parsing as URL first
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        # Check if it's a valid YouTube domain (strict matching)
        if not _is_youtube_host(hostname):
            return None

        # Handle youtu.be short URLs
        if hostname == "youtu.be" or hostname.endswith(".youtu.be"):
            # Format: https://youtu.be/VIDEO_ID
            video_id = parsed.path.lstrip("/").split("/")[0]
            if video_id and _is_valid_video_id(video_id):
                return video_id

        # Handle youtube.com URLs (hostname already validated by _is_youtube_host)
        if hostname.endswith("youtube.com") or hostname.endswith("youtube-nocookie.com"):
            path = parsed.path

            # Handle /watch URLs
            if path == "/watch" or path.startswith("/watch"):
                query_params = parse_qs(parsed.query)
                video_ids = query_params.get("v", [])
                if video_ids and _is_valid_video_id(video_ids[0]):
                    return video_ids[0]

            # Handle /shorts/ URLs
            if "/shorts/" in path:
                match = re.search(r"/shorts/([a-zA-Z0-9_-]{11})", path)
                if match:
                    return match.group(1)

            # Handle /embed/ URLs
            if "/embed/" in path:
                match = re.search(r"/embed/([a-zA-Z0-9_-]{11})", path)
                if match:
                    return match.group(1)

            # Handle /v/ URLs
            if "/v/" in path:
                match = re.search(r"/v/([a-zA-Z0-9_-]{11})", path)
                if match:
                    return match.group(1)

            # Handle /live/ URLs (YouTube live streams, including ended ones)
            if "/live/" in path:
                match = re.search(r"/live/([a-zA-Z0-9_-]{11})", path)
                if match:
                    return match.group(1)

    except Exception:
        # URL parsing failed - don't use fallback regex as it could match non-YouTube URLs
        pass

    return None


def _is_valid_video_id(video_id: str) -> bool:
    """
    Check if a string is a valid YouTube video ID.

    YouTube video IDs are 11 characters long and contain only
    alphanumeric characters, hyphens, and underscores.
    """
    if not video_id or len(video_id) != 11:
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]{11}$", video_id))


def is_youtube_url(text: str) -> bool:
    """
    Check if a text contains a YouTube URL.

    Args:
        text: The text to check

    Returns:
        True if the text contains a YouTube URL
    """
    return extract_video_id(text) is not None


def _is_bilibili_host(hostname: str) -> bool:
    """
    Check if a hostname is a valid Bilibili domain.
    """
    if not hostname:
        return False

    hostname = hostname.lower()

    for domain in BILIBILI_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return True

    return False


def extract_bilibili_video_id(url: str) -> str | None:
    """
    Extract Bilibili video ID from various URL formats.

    Supported formats:
    - https://www.bilibili.com/video/BVxxxxxxxxxx
    - https://bilibili.com/video/avxxxxxxxx
    - https://b23.tv/xxxxxxx (short URL)

    Args:
        url: The Bilibili URL to parse

    Returns:
        The video ID (BV or av number) or None if not found
    """
    if not url:
        return None

    url = url.strip()

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if not _is_bilibili_host(hostname):
            return None

        # b23.tv short URLs - return the short code
        if hostname == "b23.tv" or hostname.endswith(".b23.tv"):
            path = parsed.path.lstrip("/").split("/")[0]
            if path:
                return path
            return None

        # bilibili.com video URLs
        if hostname.endswith("bilibili.com"):
            path = parsed.path
            # Match /video/BVxxxxxxxx or /video/avxxxxxxxx
            if "/video/" in path:
                match = re.search(r"/video/(BV[a-zA-Z0-9]+|av\d+)", path)
                if match:
                    return match.group(1)

        return None

    except Exception:
        return None


def is_bilibili_url(text: str) -> bool:
    """
    Check if a text contains a Bilibili URL.

    Supported formats:
    - https://www.bilibili.com/video/BVxxxxxxxxxx
    - https://bilibili.com/video/avxxxxxxxx
    - https://b23.tv/xxxxxxx (short URL)

    Args:
        text: The text to check

    Returns:
        True if the text contains a Bilibili URL
    """
    return extract_bilibili_video_id(text) is not None


def _is_apple_podcasts_host(hostname: str) -> bool:
    """
    Check if a hostname is a valid Apple Podcasts domain.
    """
    if not hostname:
        return False

    hostname = hostname.lower()

    for domain in APPLE_PODCASTS_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return True

    return False


def extract_apple_podcasts_id(url: str) -> str | None:
    """
    Extract Apple Podcasts episode ID from URL.

    Supported formats:
    - https://podcasts.apple.com/us/podcast/xxx/id1234567890
    - https://podcasts.apple.com/us/podcast/xxx/id1234567890?i=1000xxxxxxxxx

    Args:
        url: The Apple Podcasts URL to parse

    Returns:
        A unique identifier for the episode or None if not found
    """
    if not url:
        return None

    url = url.strip()

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if not _is_apple_podcasts_host(hostname):
            return None

        path = parsed.path
        if "/podcast/" not in path:
            return None

        # Extract podcast ID (idXXXXXX)
        podcast_id_match = re.search(r"/id(\d+)", path)
        if not podcast_id_match:
            return None

        podcast_id = podcast_id_match.group(1)

        # Check for episode ID in query params (?i=1000xxxxxxxxx)
        query_params = parse_qs(parsed.query)
        episode_ids = query_params.get("i", [])
        if episode_ids:
            # Return combined ID: podcast_episode
            return f"{podcast_id}_{episode_ids[0]}"

        # Return just podcast ID if no episode specified
        return podcast_id

    except Exception:
        return None


def extract_apple_podcasts_slug(url: str) -> str | None:
    """
    Extract the URL-decoded episode (or podcast) slug from an Apple Podcasts URL.

    For URLs like /podcast/{slug}/id{podcast_id}[?i=...], the slug encodes the
    episode title (when ?i= is present) or the podcast title otherwise.
    """
    if not url:
        return None

    try:
        parsed = urlparse(url.strip())
        if not _is_apple_podcasts_host((parsed.hostname or "").lower()):
            return None

        match = re.search(r"/podcast/([^/]+)/id\d+", parsed.path)
        if not match:
            return None

        slug = unquote(match.group(1))
        return slug or None
    except Exception:
        return None


def is_apple_podcasts_url(text: str) -> bool:
    """
    Check if a text contains an Apple Podcasts URL.

    Supported formats:
    - https://podcasts.apple.com/us/podcast/xxx/id1234567890
    - https://podcasts.apple.com/us/podcast/xxx/id1234567890?i=1000xxxxxxxxx

    Args:
        text: The text to check

    Returns:
        True if the text contains an Apple Podcasts URL
    """
    return extract_apple_podcasts_id(text) is not None


def _is_xiaoyuzhou_host(hostname: str) -> bool:
    """
    Check if a hostname is a valid Xiaoyuzhou (小宇宙) domain.
    """
    if not hostname:
        return False

    hostname = hostname.lower()

    for domain in XIAOYUZHOU_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return True

    return False


def extract_xiaoyuzhou_episode_id(url: str) -> str | None:
    """
    Extract Xiaoyuzhou episode ID from URL.

    Supported formats:
    - https://www.xiaoyuzhoufm.com/episode/xxxxxxxxxxxxxxxxxxxxxxxx

    The episode ID is a 24-character hexadecimal string.

    Args:
        url: The Xiaoyuzhou URL to parse

    Returns:
        The episode ID or None if not found
    """
    if not url:
        return None

    url = url.strip()

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if not _is_xiaoyuzhou_host(hostname):
            return None

        path = parsed.path
        # Match /episode/{24-char-hex-id}
        match = re.search(r"/episode/([a-f0-9]{24})", path)
        if match:
            return match.group(1)

        return None

    except Exception:
        return None


def is_xiaoyuzhou_url(text: str) -> bool:
    """
    Check if a text contains a Xiaoyuzhou (小宇宙) podcast URL.

    Supported formats:
    - https://www.xiaoyuzhoufm.com/episode/xxxxxxxxxxxxxxxxxxxxxxxx

    Args:
        text: The text to check

    Returns:
        True if the text contains a Xiaoyuzhou URL
    """
    return extract_xiaoyuzhou_episode_id(text) is not None


def is_supported_url(text: str) -> bool:
    """
    Check if a text contains a supported URL (YouTube, Bilibili, Apple Podcasts, or Xiaoyuzhou).

    Args:
        text: The text to check

    Returns:
        True if the text contains a supported URL
    """
    return (
        is_youtube_url(text)
        or is_bilibili_url(text)
        or is_apple_podcasts_url(text)
        or is_xiaoyuzhou_url(text)
    )


def get_url_platform(text: str) -> str | None:
    """
    Get the platform name for a supported URL.

    Args:
        text: The URL text

    Returns:
        Platform name ("youtube", "bilibili", "apple_podcasts", or "xiaoyuzhou")
        or None if not supported
    """
    if is_youtube_url(text):
        return "youtube"
    if is_bilibili_url(text):
        return "bilibili"
    if is_apple_podcasts_url(text):
        return "apple_podcasts"
    if is_xiaoyuzhou_url(text):
        return "xiaoyuzhou"
    return None
