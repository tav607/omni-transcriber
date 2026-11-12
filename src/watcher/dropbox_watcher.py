"""Dropbox folder watcher using longpoll API.

Monitors a Dropbox folder for audio files and triggers transcription.
Uses Dropbox API for change detection and for reading CONTEXT.md, and rclone
for the audio and output files (download, upload, move).
"""

import asyncio
import html
import logging
import os
from typing import Set

import dropbox
from dropbox.files import FileMetadata
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from ..config import config, DropboxWatcherConfig
from .context_loader import CONTEXT_FILENAME, ContextError, load_context
from .processor import process_audio_file, is_audio_file, rclone_cat

logger = logging.getLogger(__name__)

# Audio files transcribed at once. Each already fans out to one request per
# 15-minute chunk; this keeps a batch dropped together (an earnings night)
# from starting all at the same moment.
MAX_CONCURRENT_FILES = 3
# Attempts at reading a CONTEXT.md before the file it governs is held back.
CONTEXT_READ_ATTEMPTS = 3


class DropboxWatcher:
    """
    Watches a Dropbox folder for audio files using longpoll.

    Every audio file outside the _processed / _failed subfolders is work to
    do: the ones already there at startup as well as the ones that arrive
    later. A file whose CONTEXT.md cannot be used is held back, reported, and
    tried again when a CONTEXT.md changes.
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
        # Files waiting on a CONTEXT.md fix, by path.
        self.blocked: dict[str, FileMetadata] = {}
        # asyncio keeps only weak references to tasks; an unreferenced one can
        # be collected mid-run.
        self._tasks: Set[asyncio.Task] = set()
        self._slots = asyncio.Semaphore(MAX_CONCURRENT_FILES)

    def close(self) -> None:
        """Close the Dropbox client connection."""
        if self.dbx:
            self.dbx.close()

    async def _list_folder(self) -> tuple[list[FileMetadata], str]:
        """Every file under the watch folder, and the cursor that follows it."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.dbx.files_list_folder(self.config.watch_folder, recursive=True),
        )
        entries = list(result.entries)
        while result.has_more:
            cursor = result.cursor  # Capture cursor to avoid closure issue
            result = await loop.run_in_executor(
                None,
                lambda c=cursor: self.dbx.files_list_folder_continue(c),
            )
            entries.extend(result.entries)
        return [e for e in entries if isinstance(e, FileMetadata)], result.cursor

    async def _longpoll(self, cursor: str, timeout: int = 30) -> tuple[bool, int]:
        """
        Wait for changes using Dropbox longpoll.

        Args:
            cursor: Current cursor from list_folder
            timeout: Longpoll timeout in seconds (max 480)

        Returns:
            Tuple of (changes_detected, backoff_seconds)
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.dbx.files_list_folder_longpoll(cursor, timeout=timeout),
        )
        return result.changes, result.backoff if result.backoff else 0

    async def _get_changes(self, cursor: str) -> tuple[list[FileMetadata], str]:
        """
        Get file changes since the last cursor.

        Args:
            cursor: Previous cursor

        Returns:
            Tuple of (list of new/changed file metadata, new cursor)
        """
        loop = asyncio.get_running_loop()
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

    async def _read_file(self, path: str) -> str | None:
        """Text of a small Dropbox file: None when there is no such file,
        ContextError when that could not be settled after a few attempts. The
        difference matters: a CONTEXT.md that failed to download must not be
        taken for a folder without one.

        Whether the file exists comes from the Dropbox API (the app's token
        reads metadata only); its content comes through rclone, like every
        other file the watcher touches.
        """
        loop = asyncio.get_running_loop()
        base = self.config.rclone_base_path.rstrip("/")
        last_error = "unknown error"
        for attempt in range(1, CONTEXT_READ_ATTEMPTS + 1):
            try:
                await loop.run_in_executor(None, lambda: self.dbx.files_get_metadata(path))
            except dropbox.exceptions.ApiError as error:
                lookup = error.error.get_path() if error.error.is_path() else None
                if lookup is not None and lookup.is_not_found():
                    return None
                last_error = str(error)
            except Exception as error:  # network trouble
                last_error = str(error)
            else:
                content = await rclone_cat(f"{self.config.rclone_remote}:{base}{path}")
                if content is not None:
                    return content
                last_error = "rclone could not download it"
            if attempt < CONTEXT_READ_ATTEMPTS:
                await asyncio.sleep(2 * attempt)
        raise ContextError(f"{path}: could not be read from Dropbox ({last_error})")

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

        # Skip files already handled, one way or the other
        for subfolder in (self.config.processed_subfolder, self.config.failed_subfolder):
            if f"/{subfolder}/" in path:
                return False

        # Skip non-audio files
        if not is_audio_file(filename):
            return False

        # Skip files currently being processed
        if path in self.processing_files:
            return False

        return True

    async def _notify(self, text: str) -> None:
        if not self.config.telegram_chat_id:
            return
        try:
            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id, text=text, parse_mode="HTML")
        except Exception as e:
            logger.warning(f"Failed to send watcher notification: {e}")

    async def _process_file(self, file_metadata: FileMetadata) -> None:
        """
        Process a single file.

        Args:
            file_metadata: Dropbox file metadata
        """
        path = file_metadata.path_display or file_metadata.path_lower
        file_size = file_metadata.size if hasattr(file_metadata, 'size') else 0
        modified = getattr(file_metadata, "client_modified", None)

        try:
            async with self._slots:
                logger.info(f"Processing audio file: {path} ({file_size} bytes)")

                try:
                    context = await load_context(path, self.config.watch_folder, self._read_file)
                except ContextError as error:
                    # Held back, not failed: the audio is fine, its settings are not.
                    logger.error(f"CONTEXT.md problem, holding back {path}: {error}")
                    self.blocked[path] = file_metadata
                    await self._notify(
                        f"⚠️ Not processed, CONTEXT.md needs fixing:\n"
                        f"<code>{html.escape(path.lstrip('/'))}</code>\n\n"
                        f"{html.escape(str(error))}\n\n"
                        "The file stays where it is and runs once a CONTEXT.md is saved again."
                    )
                    return
                self.blocked.pop(path, None)

                # Process the audio file
                from ..config import config as app_config
                await process_audio_file(
                    dropbox_path=path,
                    context=context,
                    bot=self.bot,
                    config=app_config,
                    file_size_bytes=file_size,
                    client_modified=modified.date() if modified else None,
                )

        except Exception as e:
            logger.error(f"Error processing {path}: {e}", exc_info=True)

        finally:
            self.processing_files.discard(path)

    def _schedule(self, files: list[FileMetadata]) -> int:
        """Start a background task for each file that is work to do."""
        started = 0
        for f in files:
            if not self._should_process_file(f):
                continue
            # Marked here, not inside the task, so a second event for the same
            # file cannot start it twice before the first task runs.
            self.processing_files.add(f.path_display or f.path_lower)
            task = asyncio.create_task(self._process_file(f))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            started += 1
        return started

    async def _start_from_listing(self) -> None:
        """Take a fresh cursor and pick up every audio file waiting in the
        folder: ones dropped while the service was down, and ones held back
        or left behind by an earlier run."""
        files, self.cursor = await self._list_folder()
        started = self._schedule(files)
        if started:
            logger.info(f"Found {started} audio file(s) waiting in the watch folder")

    async def watch(self) -> None:
        """
        Main watch loop using longpoll.

        Runs indefinitely, polling Dropbox for changes and processing
        audio files as they appear.
        """
        logger.info(f"Starting Dropbox watcher for: {self.config.watch_folder}")

        await self._start_from_listing()
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

                # A saved CONTEXT.md is the cue to retry what was held back on one.
                if self.blocked and any(
                    os.path.basename(f.path_display or f.path_lower) == CONTEXT_FILENAME
                    for f in new_files
                ):
                    retry = list(self.blocked.values())
                    self.blocked.clear()
                    logger.info(f"A CONTEXT.md changed, retrying {len(retry)} held-back file(s)")
                    new_files = new_files + retry

                started = self._schedule(new_files)
                if started:
                    logger.info(f"Detected {started} new audio file(s)")

            except dropbox.exceptions.AuthError as e:
                logger.error(f"Dropbox auth error: {e}")
                logger.error("Please check your DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN")
                # Don't retry auth errors, exit the loop
                raise

            except dropbox.exceptions.ApiError as e:
                # A cursor Dropbox no longer accepts never recovers by itself:
                # list the folder again, which also catches what was missed.
                logger.error(f"Dropbox API error in watch loop, taking a fresh cursor: {e}")
                await asyncio.sleep(self.config.poll_interval)
                try:
                    await self._start_from_listing()
                except Exception as error:
                    logger.error(f"Could not re-list the watch folder: {error}")

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
