"""CONTEXT.md loader for Dropbox watcher.

Loads context configuration from CONTEXT.md files in Dropbox directories.
Searches from the audio file's directory up to the watch folder root.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


def _is_safe_dropbox_path(path: str, allow_empty: bool = False) -> bool:
    """
    Check if a Dropbox path is safe (no path traversal).

    Must be called BEFORE os.normpath, as normpath removes '..' sequences.

    Args:
        path: Dropbox path to check
        allow_empty: If True, allow empty string (for App folder root)

    Returns:
        True if path is safe, False otherwise.
    """
    # Empty string is valid for App folder root if allowed
    if not path:
        return allow_empty

    # Check for path traversal attempts before normalization
    # Use PurePosixPath since Dropbox paths are always POSIX-style
    if ".." in PurePosixPath(path).parts:
        return False

    return True


@dataclass
class ContextConfig:
    """Configuration loaded from CONTEXT.md file."""

    background: str = ""  # Background information about the recordings
    sections: str = ""  # Custom section definitions for JSON output format
    extra_fields: dict = field(default_factory=dict)  # Any additional fields


async def rclone_cat(path: str) -> str | None:
    """
    Read a remote file's content using rclone cat.

    Args:
        path: rclone path like "dropbox:/path/to/file"

    Returns:
        File content as string, or None if file doesn't exist or read fails.
    """
    proc = await asyncio.create_subprocess_exec(
        "rclone", "cat", path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        return stdout.decode("utf-8")

    # File doesn't exist or other error
    return None


def parse_context_md(content: str) -> ContextConfig:
    """
    Parse CONTEXT.md content into ContextConfig.

    Expected format:
    ```markdown
    # Context

    ## Background
    Some background info about the recordings in this folder.

    ## Sections
    Custom section definitions for JSON output format.
    Defines what fields the LLM should output (title, summary, key_points, etc.)
    ```

    Args:
        content: The raw CONTEXT.md content

    Returns:
        Parsed ContextConfig
    """
    config = ContextConfig()

    if not content:
        return config

    # Split into sections by h2 headers
    sections = re.split(r'^##\s+', content, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        # Split section into header and body
        lines = section.strip().split('\n', 1)
        if not lines:
            continue

        header = lines[0].strip().lower()
        body = lines[1].strip() if len(lines) > 1 else ""

        if header == "background":
            config.background = body
        elif header in ("sections", "output format", "output_format"):
            config.sections = body
        else:
            # Store any other sections in extra_fields
            config.extra_fields[header] = body

    return config


async def load_context(
    dropbox_audio_path: str,
    watch_folder: str,
    rclone_remote: str = "dropbox",
    rclone_base_path: str = "",
) -> ContextConfig:
    """
    Load CONTEXT.md from the audio file's directory, searching up to watch folder.

    Searches from the audio file's parent directory upward until finding a
    CONTEXT.md or reaching the watch folder root.

    Args:
        dropbox_audio_path: Dropbox path to the audio file, e.g., "/Recordings/meetings/file.m4a"
        watch_folder: The root watch folder, e.g., "/Recordings"
        rclone_remote: rclone remote name (default: "dropbox")
        rclone_base_path: Base path for rclone (maps App folder to full Dropbox path)

    Returns:
        ContextConfig parsed from CONTEXT.md, or empty config if not found.
    """
    # Validate paths for traversal attacks BEFORE normpath (which removes '..')
    if not _is_safe_dropbox_path(dropbox_audio_path):
        logger.warning(f"Unsafe audio path rejected: {dropbox_audio_path!r}")
        return ContextConfig()

    if not _is_safe_dropbox_path(watch_folder, allow_empty=True):
        logger.warning(f"Unsafe watch folder rejected: {watch_folder!r}")
        return ContextConfig()

    # Normalize paths after validation
    dropbox_audio_path = os.path.normpath(dropbox_audio_path)

    # Special handling for empty watch_folder (App folder root)
    # Empty string means all paths are within watch scope
    is_app_folder_root = (watch_folder == "")
    if not is_app_folder_root:
        watch_folder = os.path.normpath(watch_folder)
        # Ensure watch_folder ends with separator for proper prefix matching
        watch_folder_prefix = watch_folder.rstrip("/") + "/"

    # Normalize rclone_base_path
    rclone_base_path = rclone_base_path.rstrip("/")

    def to_rclone_path(sdk_path: str) -> str:
        """Convert Dropbox SDK path to rclone full path."""
        if rclone_base_path:
            return f"{rclone_remote}:{rclone_base_path}{sdk_path}"
        return f"{rclone_remote}:{sdk_path}"

    # Start from the audio file's directory
    current_dir = os.path.dirname(dropbox_audio_path)

    while current_dir:
        # Normalize again after dirname
        current_dir = os.path.normpath(current_dir)

        # Check if we've gone above the watch folder
        # Skip this check for App folder root (all paths are valid)
        if not is_app_folder_root:
            # Use proper prefix check: current_dir must be watch_folder or start with watch_folder/
            if current_dir != watch_folder.rstrip("/") and not current_dir.startswith(watch_folder_prefix):
                break

        context_path = f"{current_dir}/CONTEXT.md"
        rclone_path = to_rclone_path(context_path)

        logger.debug(f"Looking for CONTEXT.md at: {rclone_path}")
        content = await rclone_cat(rclone_path)

        if content:
            logger.info(f"Found CONTEXT.md at: {context_path}")
            return parse_context_md(content)

        # Move up one directory
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            # Reached root
            break
        current_dir = parent

    logger.debug(f"No CONTEXT.md found for: {dropbox_audio_path}")
    return ContextConfig()
