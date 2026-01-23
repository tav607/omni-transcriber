"""Dropbox folder watcher using longpoll API.

Monitors a Dropbox folder for new audio files and triggers transcription.
Uses Dropbox API for change detection and rclone for file operations.
"""

import asyncio
import logging
import os
from typing import Set

import dropbox
from dropbox.files import FileMetadata

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from ..config import config, DropboxWatcherConfig
from .context_loader import load_context
from .processor import process_audio_file, is_audio_file

logger = logging.getLogger(__name__)


class DropboxWatcher:
    """
    Watches a Dropbox folder for new audio files using longpoll.

    Uses Dropbox API's list_folder/longpoll for efficient change detection,
    and rclone for all file operations (download, upload, move).
    """

    def __init__(self, watcher_config: DropboxWatcherConfig, bot: Bot):
        """
        Initialize the watcher.

        Args:
            watcher_config: Dropbox watcher configuration
            bot: Telegram bot instance for notifications
        """
        self.config = watcher_config
        self.bot = bot
        self.dbx = dropbox.Dropbox(
            app_key=watcher_config.app_key,
            app_secret=watcher_config.app_secret,
            oauth2_refresh_token=watcher_config.refresh_token,
        )
        self.cursor: str | None = None
        self.processing_files: Set[str] = set()  # Track files being processed

    def close(self) -> None:
        """Close the Dropbox client connection."""
        if self.dbx:
            self.dbx.close()

    async def _get_initial_cursor(self) -> str:
        """
        Get the initial cursor for the watch folder.

        Returns:
            Cursor string for subsequent longpoll calls.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.dbx.files_list_folder(
                self.config.watch_folder,
                recursive=True,
            ),
        )

        # Process initial entries to handle has_more pagination
        while result.has_more:
            cursor = result.cursor  # Capture cursor to avoid closure issue
            result = await loop.run_in_executor(
                None,
                lambda c=cursor: self.dbx.files_list_folder_continue(c),
            )

        return result.cursor

    async def _longpoll(self, cursor: str, timeout: int = 30) -> tuple[bool, int]:
        """
        Wait for changes using Dropbox longpoll.

        Args:
            cursor: Current cursor from list_folder
            timeout: Longpoll timeout in seconds (max 480)

        Returns:
            Tuple of (changes_detected, backoff_seconds)
        """
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.dbx.files_list_folder_longpoll(cursor, timeout=timeout),
            )
            return result.changes, result.backoff if result.backoff else 0
        except dropbox.exceptions.ApiError as e:
            logger.error(f"Longpoll API error: {e}")
            return False, 30  # Backoff on error

    async def _get_changes(self, cursor: str) -> tuple[list[FileMetadata], str]:
        """
        Get file changes since the last cursor.

        Args:
            cursor: Previous cursor

        Returns:
            Tuple of (list of new/changed file metadata, new cursor)
        """
        loop = asyncio.get_event_loop()
        all_entries = []

        result = await loop.run_in_executor(
            None,
            lambda: self.dbx.files_list_folder_continue(cursor),
        )

        all_entries.extend(result.entries)

        # Handle pagination
        while result.has_more:
            cursor = result.cursor  # Capture cursor to avoid closure issue
            result = await loop.run_in_executor(
                None,
                lambda c=cursor: self.dbx.files_list_folder_continue(c),
            )
            all_entries.extend(result.entries)

        # Filter for new files only (not folders or deletions)
        new_files = [
            entry for entry in all_entries
            if isinstance(entry, FileMetadata)
        ]

        return new_files, result.cursor

    def _should_process_file(self, file_metadata: FileMetadata) -> bool:
        """
        Check if a file should be processed.

        Args:
            file_metadata: Dropbox file metadata

        Returns:
            True if the file should be processed.
        """
        path = file_metadata.path_display or file_metadata.path_lower
        filename = os.path.basename(path)

        # Skip files in _processed directories
        if f"/{self.config.processed_subfolder}/" in path:
            return False

        # Skip non-audio files
        if not is_audio_file(filename):
            return False

        # Skip files currently being processed
        if path in self.processing_files:
            return False

        return True

    async def _process_file(self, file_metadata: FileMetadata) -> None:
        """
        Process a single file.

        Args:
            file_metadata: Dropbox file metadata
        """
        path = file_metadata.path_display or file_metadata.path_lower
        file_size = file_metadata.size if hasattr(file_metadata, 'size') else 0
        self.processing_files.add(path)

        try:
            logger.info(f"Processing new audio file: {path} ({file_size} bytes)")

            # Load context from CONTEXT.md
            context = await load_context(
                path,
                self.config.watch_folder,
                self.config.rclone_remote,
            )
            if context.sections:
                logger.info("Using custom sections from CONTEXT.md")

            # Process the audio file
            from ..config import config as app_config
            await process_audio_file(
                dropbox_path=path,
                context=context,
                bot=self.bot,
                config=app_config,
                file_size_bytes=file_size,
            )

        except Exception as e:
            logger.error(f"Error processing {path}: {e}", exc_info=True)
        finally:
            self.processing_files.discard(path)

    async def watch(self) -> None:
        """
        Main watch loop using longpoll.

        Runs indefinitely, polling Dropbox for changes and processing
        new audio files as they appear.
        """
        logger.info(f"Starting Dropbox watcher for: {self.config.watch_folder}")

        # Get initial cursor
        self.cursor = await self._get_initial_cursor()
        logger.info("Got initial cursor, starting longpoll loop")

        while True:
            try:
                # Wait for changes
                changes_detected, backoff = await self._longpoll(
                    self.cursor,
                    timeout=self.config.poll_interval,
                )

                if backoff > 0:
                    logger.info(f"Dropbox requested backoff: {backoff}s")
                    await asyncio.sleep(backoff)
                    continue

                if not changes_detected:
                    # Timeout with no changes, continue polling
                    continue

                # Get the actual changes
                new_files, new_cursor = await self._get_changes(self.cursor)
                self.cursor = new_cursor

                # Filter files to process
                files_to_process = [
                    f for f in new_files
                    if self._should_process_file(f)
                ]

                if not files_to_process:
                    continue

                logger.info(f"Detected {len(files_to_process)} new audio file(s)")

                # Process files sequentially to avoid overwhelming the system
                for file_metadata in files_to_process:
                    await self._process_file(file_metadata)

            except dropbox.exceptions.AuthError as e:
                logger.error(f"Dropbox auth error: {e}")
                logger.error("Please check your DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN")
                # Don't retry auth errors, exit the loop
                raise

            except Exception as e:
                logger.error(f"Error in watch loop: {e}", exc_info=True)
                # Wait before retrying
                await asyncio.sleep(self.config.poll_interval)


def create_bot_for_watcher() -> Bot:
    """
    Create a minimal bot instance for sending notifications.

    Uses the same token as the main bot but without polling setup.
    """
    from ..bot.bot import get_proxy_url
    from aiogram.client.session.aiohttp import AiohttpSession

    proxy_url = get_proxy_url()
    session = None

    if proxy_url:
        session = AiohttpSession(proxy=proxy_url)

    return Bot(
        token=config.telegram.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        session=session,
    )


async def run_dropbox_watcher() -> None:
    """
    Run the Dropbox watcher.

    This is the main entry point called from main.py.
    """
    watcher_config = config.dropbox_watcher

    if not watcher_config.is_enabled:
        logger.info("Dropbox watcher is disabled (missing config)")
        return

    logger.info("Initializing Dropbox watcher...")

    # Create a bot instance for notifications
    bot = create_bot_for_watcher()
    watcher = None

    try:
        watcher = DropboxWatcher(watcher_config, bot)
        await watcher.watch()
    finally:
        if watcher:
            watcher.close()
        if bot.session:
            await bot.session.close()
