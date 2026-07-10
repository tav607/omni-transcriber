"""RSS feed parser for Apple Podcasts metadata extraction."""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp
import feedparser
import requests

from ..utils.url_parser import extract_apple_podcasts_slug

logger = logging.getLogger(__name__)

# Browser-like UA: feed.xyzfm.space (Xiaoyuzhou's feed host, common behind
# Apple feedUrls for Chinese podcasts) returns 403 for bot-style UAs since 2026-06.
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def _fetch_text_sync(url: str, timeout: int = 60) -> Optional[str]:
    """Synchronous fallback fetch via requests.

    Some feed hosts (anchor.fm, acast.com) stall aiohttp until timeout but
    respond fine to requests.
    """
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT})
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Sync fetch failed for {url}: {e}")
        return None


@dataclass
class EpisodeMetadata:
    """Metadata about a podcast episode."""
    podcast_name: str
    episode_title: str
    episode_link: str = ""
    publish_date: str = ""
    shownotes: str = ""  # Episode description for context
    audio_url: str = ""


def extract_podcast_id(apple_url: str) -> Optional[str]:
    """
    Extract podcast ID from Apple Podcast URL.

    Examples:
        https://podcasts.apple.com/cn/podcast/硅谷101/id1498541229
        https://podcasts.apple.com/us/podcast/lex-fridman-podcast/id1434243584
        https://podcasts.apple.com/us/podcast/.../id1434243584?i=1000123456789
    """
    match = re.search(r'/id(\d+)', apple_url)
    if match:
        return match.group(1)
    return None


def _normalize_title(s: str) -> str:
    """Lowercase and strip non-alphanumeric chars (Unicode word chars only)."""
    if not s:
        return ""
    return re.sub(r"\W+", "", s.lower(), flags=re.UNICODE)


def extract_episode_id(apple_url: str) -> Optional[str]:
    """
    Extract episode ID from Apple Podcast URL (if present).

    Example:
        https://podcasts.apple.com/us/podcast/.../id1434243584?i=1000123456789
        -> episode_id = 1000123456789
    """
    match = re.search(r'[?&]i=(\d+)', apple_url)
    if match:
        return match.group(1)
    return None


async def get_feed_url(podcast_id: str) -> Optional[str]:
    """
    Get RSS feed URL from Apple iTunes API.

    Args:
        podcast_id: The numeric Apple Podcast ID

    Returns:
        RSS feed URL if found, None otherwise
    """
    lookup_url = f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast"

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(lookup_url) as response:
                if response.status != 200:
                    logger.error(f"iTunes API returned status {response.status}")
                    return None

                data = await response.json(content_type=None)

                if data.get("resultCount", 0) == 0:
                    logger.warning(f"No podcast found for ID {podcast_id}")
                    return None

                result = data["results"][0]
                feed_url = result.get("feedUrl")

                if feed_url:
                    logger.debug(f"Found feed URL for podcast {podcast_id}: {feed_url}")
                    return feed_url
                else:
                    logger.warning(f"No feedUrl in response for podcast {podcast_id}")
                    return None

    except asyncio.TimeoutError:
        logger.warning(f"aiohttp timeout for iTunes API, falling back to requests")
    except aiohttp.ClientError as e:
        logger.warning(f"aiohttp error for iTunes API: {e}, falling back to requests")

    text = await asyncio.get_running_loop().run_in_executor(None, _fetch_text_sync, lookup_url)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"iTunes API fallback returned invalid JSON: {e}")
        return None
    if data.get("resultCount", 0) == 0:
        logger.warning(f"No podcast found for ID {podcast_id}")
        return None
    feed_url = data["results"][0].get("feedUrl")
    if not feed_url:
        logger.warning(f"No feedUrl in response for podcast {podcast_id}")
        return None
    return feed_url


async def fetch_feed_content(feed_url: str) -> Optional[str]:
    """Fetch RSS feed content."""
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(feed_url) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for {feed_url}")
                    return None
                return await response.text()

    except asyncio.TimeoutError:
        logger.warning(f"aiohttp timeout for {feed_url}, falling back to requests")
    except aiohttp.ClientError as e:
        logger.warning(f"aiohttp error for {feed_url}: {e}, falling back to requests")

    return await asyncio.get_running_loop().run_in_executor(None, _fetch_text_sync, feed_url)


async def get_episode_metadata(apple_url: str) -> Optional[EpisodeMetadata]:
    """
    Get episode metadata from an Apple Podcast URL.

    This function:
    1. Extracts podcast ID from URL
    2. Gets RSS feed URL from iTunes API
    3. Parses RSS feed to find the specific episode
    4. Returns metadata for the episode

    Args:
        apple_url: Apple Podcast episode URL

    Returns:
        EpisodeMetadata if found, None otherwise
    """
    # Extract podcast ID
    podcast_id = extract_podcast_id(apple_url)
    if not podcast_id:
        logger.error(f"Could not extract podcast ID from URL: {apple_url}")
        return None

    target_episode_id = extract_episode_id(apple_url)
    target_slug_norm = ""
    if target_episode_id:
        target_slug_norm = _normalize_title(extract_apple_podcasts_slug(apple_url) or "")

    # Get feed URL
    feed_url = await get_feed_url(podcast_id)
    if not feed_url:
        logger.error(f"Could not get feed URL for podcast {podcast_id}")
        return None

    # Fetch and parse feed
    content = await fetch_feed_content(feed_url)
    if not content:
        logger.error(f"Failed to fetch feed {feed_url}")
        return None

    try:
        feed = feedparser.parse(content)

        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

        podcast_name = feed.feed.get("title", "Unknown Podcast")

        target_entry = None

        if target_episode_id:
            # Apple's iTunes episode ID often isn't in the feed's guid/id (e.g.
            # Ximalaya-hosted feeds use their own ID space), so fall back to
            # matching the URL slug against entry titles.
            slug_match_enabled = bool(target_slug_norm) and len(target_slug_norm) >= 6
            id_match = exact = contains = None

            for entry in feed.entries:
                if id_match is None and any(
                    target_episode_id in str(entry.get(k, "")) for k in ("id", "guid")
                ):
                    id_match = entry
                    break
                if slug_match_enabled and exact is None:
                    n = _normalize_title(entry.get("title", ""))
                    if not n:
                        continue
                    if n == target_slug_norm:
                        exact = entry
                    elif contains is None and target_slug_norm in n:
                        contains = entry

            target_entry = id_match or exact or contains

            if target_entry is None:
                logger.error(
                    f"Could not locate episode {target_episode_id} (slug={target_slug_norm!r}) in feed {feed_url}"
                )
                return None
        elif feed.entries:
            target_entry = feed.entries[0]

        if target_entry is None:
            logger.error("No episodes found in feed")
            return None

        # Extract audio URL
        audio_url = ""
        for enclosure in target_entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href") or enclosure.get("url", "")
                break

        # Parse published date
        publish_date = ""
        try:
            if hasattr(target_entry, "published_parsed") and target_entry.published_parsed:
                dt = datetime(*target_entry.published_parsed[:6])
                publish_date = dt.strftime("%Y-%m-%d")
            elif hasattr(target_entry, "updated_parsed") and target_entry.updated_parsed:
                dt = datetime(*target_entry.updated_parsed[:6])
                publish_date = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

        # Get description/shownotes
        shownotes = (
            target_entry.get("summary") or
            target_entry.get("description") or
            (target_entry.get("content", [{}])[0].get("value", "") if target_entry.get("content") else "")
        )

        # Clean HTML from shownotes
        shownotes = re.sub(r"<[^>]+>", "", shownotes)
        shownotes = shownotes.strip()[:2000]  # Limit length

        episode_title = target_entry.get("title", "Untitled Episode")

        metadata = EpisodeMetadata(
            podcast_name=podcast_name,
            episode_title=episode_title,
            episode_link=apple_url,
            publish_date=publish_date,
            shownotes=shownotes,
            audio_url=audio_url,
        )

        logger.info(f"Got metadata for episode: {podcast_name} - {episode_title}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to parse feed: {e}")
        return None
