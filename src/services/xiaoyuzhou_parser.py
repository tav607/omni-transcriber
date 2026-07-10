"""Xiaoyuzhou (小宇宙) podcast metadata extraction via RSSHub."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp
import feedparser
import requests

from ..config import config

logger = logging.getLogger(__name__)

# Browser-like UA: xiaoyuzhoufm.com and feed.xyzfm.space return 403 for
# bot-style UAs since 2026-06.
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


@dataclass
class XiaoyuzhouEpisodeMetadata:
    """Metadata about a Xiaoyuzhou podcast episode."""

    podcast_name: str
    episode_title: str
    episode_link: str = ""
    publish_date: str = ""
    shownotes: str = ""
    audio_url: str = ""


def _fetch_url_sync(url: str, timeout: int = 60) -> Optional[str]:
    """Synchronous fallback fetch via requests (some hosts stall aiohttp)."""
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT})
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Sync fetch failed for {url}: {e}")
        return None


async def _fetch_url(url: str, timeout_seconds: int = 30) -> Optional[str]:
    """Fetch URL content with aiohttp, falling back to requests if needed."""
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for {url}")
                    return None
                return await response.text()

    except asyncio.TimeoutError:
        logger.warning(f"aiohttp timeout for {url}, falling back to requests")
    except aiohttp.ClientError as e:
        logger.warning(f"aiohttp error for {url}: {e}, falling back to requests")

    return await asyncio.get_running_loop().run_in_executor(None, _fetch_url_sync, url)


async def get_podcast_id_from_episode(episode_id: str) -> Optional[str]:
    """
    Fetch episode page from xiaoyuzhou and extract the podcast_id.

    The episode page HTML contains links to the podcast page.
    We try multiple strategies to extract the podcast_id.

    Args:
        episode_id: The episode ID (24-char hex)

    Returns:
        podcast_id or None if extraction fails
    """
    episode_url = f"https://www.xiaoyuzhoufm.com/episode/{episode_id}"

    html = await _fetch_url(episode_url)
    if not html:
        return None

    # Strategy 1: Look for href="/podcast/{podcast_id}" in HTML
    match = re.search(r'href="/podcast/([a-f0-9]{24})"', html)
    if match:
        podcast_id = match.group(1)
        logger.debug(f"Found podcast_id via href pattern: {podcast_id}")
        return podcast_id

    # Strategy 2: Parse __NEXT_DATA__ JSON
    next_data_match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            # Navigate to episode.podcast.id (structure may vary)
            props = data.get("props", {})
            page_props = props.get("pageProps", {})
            episode_data = page_props.get("episode", {})
            podcast_data = episode_data.get("podcast", {})
            podcast_id = podcast_data.get("id")
            if podcast_id:
                logger.debug(f"Found podcast_id via __NEXT_DATA__: {podcast_id}")
                return podcast_id
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse __NEXT_DATA__: {e}")

    # Strategy 3: Look for other patterns
    # Sometimes the podcast link might be in a different format
    match = re.search(r'"podcastId"\s*:\s*"([a-f0-9]{24})"', html)
    if match:
        podcast_id = match.group(1)
        logger.debug(f"Found podcast_id via podcastId pattern: {podcast_id}")
        return podcast_id

    logger.error(f"Could not extract podcast_id from episode page: {episode_url}")
    return None


async def get_episode_metadata(episode_id: str) -> Optional[XiaoyuzhouEpisodeMetadata]:
    """
    Get episode metadata from a Xiaoyuzhou episode URL via RSSHub.

    This function:
    1. Fetches episode page to extract podcast_id
    2. Builds RSSHub URL for the podcast RSS feed
    3. Parses RSS feed to find the specific episode
    4. Returns metadata including audio_url

    Args:
        episode_id: Xiaoyuzhou episode ID (24-char hex)

    Returns:
        XiaoyuzhouEpisodeMetadata if found, None otherwise
    """
    # Check if RSSHub is configured
    if not config.rsshub.is_enabled:
        logger.error(
            "RSSHub is not configured. Set RSSHUB_BASE_URL environment variable "
            "to enable Xiaoyuzhou podcast support."
        )
        return None

    # Step 1: Get podcast_id from episode page
    podcast_id = await get_podcast_id_from_episode(episode_id)
    if not podcast_id:
        logger.error(f"Could not get podcast_id for episode {episode_id}")
        return None

    logger.info(f"Found podcast_id: {podcast_id} for episode: {episode_id}")

    # Step 2: Build RSSHub URL
    rsshub_url = f"{config.rsshub.base_url}/xiaoyuzhou/podcast/{podcast_id}"
    if config.rsshub.key:
        rsshub_url += f"?key={config.rsshub.key}"

    logger.debug(f"Fetching RSS from: {rsshub_url}")

    # Step 3: Fetch and parse RSS feed
    content = await _fetch_url(rsshub_url, timeout_seconds=60)
    if not content:
        logger.error(f"Failed to fetch RSSHub feed for podcast {podcast_id}")
        return None

    try:
        feed = feedparser.parse(content)

        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

        podcast_name = feed.feed.get("title", "Unknown Podcast")

        # Find the target episode by matching episode_id in guid/link
        target_entry = None
        episode_url = f"https://www.xiaoyuzhoufm.com/episode/{episode_id}"

        for entry in feed.entries:
            entry_link = entry.get("link", "")
            entry_id = str(entry.get("id", "") or entry.get("guid", ""))

            # Match by episode_id in link or guid
            if episode_id in entry_link or episode_id in entry_id:
                target_entry = entry
                logger.debug(f"Found matching episode in RSS feed")
                break

        if target_entry is None:
            logger.error(
                f"Episode {episode_id} not found in RSS feed. "
                "The episode may be too old or the RSS feed may not include it."
            )
            return None

        # Extract audio URL from enclosure
        audio_url = ""
        for enclosure in target_entry.get("enclosures", []):
            enc_type = enclosure.get("type", "")
            if enc_type.startswith("audio/") or enc_type == "":
                audio_url = enclosure.get("href") or enclosure.get("url", "")
                if audio_url:
                    break

        # Fallback to links
        if not audio_url:
            for link in target_entry.get("links", []):
                link_type = link.get("type", "")
                if link_type.startswith("audio/"):
                    audio_url = link.get("href", "")
                    break

        if not audio_url:
            logger.error("No audio URL found in episode enclosure")
            return None

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
            target_entry.get("summary")
            or target_entry.get("description")
            or (
                target_entry.get("content", [{}])[0].get("value", "")
                if target_entry.get("content")
                else ""
            )
        )

        # Clean HTML from shownotes
        shownotes = re.sub(r"<[^>]+>", "", shownotes)
        shownotes = shownotes.strip()[:2000]  # Limit length

        episode_title = target_entry.get("title", "Untitled Episode")

        metadata = XiaoyuzhouEpisodeMetadata(
            podcast_name=podcast_name,
            episode_title=episode_title,
            episode_link=episode_url,
            publish_date=publish_date,
            shownotes=shownotes,
            audio_url=audio_url,
        )

        logger.info(f"Got xiaoyuzhou metadata: {podcast_name} - {episode_title}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to parse RSS feed: {e}")
        return None
