"""Audio processor for Dropbox watcher.

Handles the transcription and editing workflow for audio files
detected by the Dropbox watcher.
"""

import asyncio
import html
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import PurePosixPath

from aiogram import Bot
from aiogram.types import FSInputFile

from ..config import AppConfig, DropboxWatcherConfig
from ..services.transcriber import transcribe, TranscriptionMetadata
from ..services.editor import edit
from ..services.pdf_generator import generate_pdf
from .context_loader import ContextConfig

logger = logging.getLogger(__name__)

# Pattern to extract h1 title from markdown
H1_TITLE_PATTERN = re.compile(r"^#\s+(.+?)$", re.MULTILINE)

# Supported audio extensions
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".ogg", ".flac", ".aac"}

# Pattern for valid Dropbox paths (no control characters, no '..' traversal)
VALID_PATH_PATTERN = re.compile(r"^/[^\x00-\x1f]*$")


def is_audio_file(filename: str) -> bool:
    """Check if a filename has a supported audio extension."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in AUDIO_EXTENSIONS


def validate_dropbox_path(path: str) -> bool:
    """
    Validate a Dropbox path for safety.

    Args:
        path: Dropbox path to validate

    Returns:
        True if path is safe, False otherwise.
    """
    if not path:
        return False

    # Must start with /
    if not path.startswith("/"):
        return False

    # Check for control characters
    if not VALID_PATH_PATTERN.match(path):
        return False

    # Check for path traversal attempts
    # Must check original path parts BEFORE normalization, as normpath removes '..'
    if ".." in PurePosixPath(path).parts:
        return False

    return True


def extract_title_from_transcript(transcript: str) -> str | None:
    """Extract the h1 title from a markdown transcript."""
    match = H1_TITLE_PATTERN.search(transcript)
    if match:
        return match.group(1).strip()
    return None


def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """Sanitize a filename by removing illegal characters."""
    filename = os.path.basename(filename)
    name, ext = os.path.splitext(filename)

    illegal_chars = r'[\\/:*?"<>|\x00-\x1f]'
    name = re.sub(illegal_chars, '', name)
    ext = re.sub(illegal_chars, '', ext)
    name = re.sub(r'\s+', ' ', name).strip()

    if len(name) > max_length:
        name = name[:max_length].strip()

    if not name:
        name = "file"

    return name + ext


async def rclone_copy(src: str, dst: str) -> bool:
    """
    Copy a file using rclone.

    Args:
        src: Source path (can be local or remote like "dropbox:/path")
        dst: Destination path (can be local or remote)

    Returns:
        True if copy succeeded, False otherwise.
    """
    logger.info(f"rclone copy: {src} -> {dst}")
    proc = await asyncio.create_subprocess_exec(
        "rclone", "copy", src, dst,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(f"rclone copy failed: {stderr.decode()}")
        return False

    return True


async def rclone_copyto(src: str, dst: str) -> bool:
    """
    Copy a file to a specific destination path using rclone copyto.

    Args:
        src: Source file path
        dst: Destination file path (not directory)

    Returns:
        True if copy succeeded, False otherwise.
    """
    logger.info(f"rclone copyto: {src} -> {dst}")
    proc = await asyncio.create_subprocess_exec(
        "rclone", "copyto", src, dst,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(f"rclone copyto failed: {stderr.decode()}")
        return False

    return True


async def rclone_moveto(src: str, dst: str) -> bool:
    """
    Move a file using rclone moveto.

    Args:
        src: Source path (remote like "dropbox:/path/file")
        dst: Destination path (remote like "dropbox:/path/_processed/file")

    Returns:
        True if move succeeded, False otherwise.
    """
    logger.info(f"rclone moveto: {src} -> {dst}")
    proc = await asyncio.create_subprocess_exec(
        "rclone", "moveto", src, dst,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(f"rclone moveto failed: {stderr.decode()}")
        return False

    return True


async def process_audio_file(
    dropbox_path: str,
    context: ContextConfig,
    bot: Bot,
    config: AppConfig,
    file_size_bytes: int = 0,
) -> bool:
    """
    Process an audio file from Dropbox.

    1. Download audio from Dropbox using rclone
    2. Transcribe using Gemini
    3. Edit/format using Gemini (with optional CONTEXT.md prompt override)
    4. Generate PDF
    5. Upload results to Dropbox output folder
    6. Send Telegram notification
    7. Move original audio to _processed subfolder
    8. Clean up local temp files

    Args:
        dropbox_path: Dropbox path to the audio file, e.g., "/Recordings/file.m4a"
        context: ContextConfig loaded from CONTEXT.md
        bot: Telegram bot instance for notifications
        config: Application config
        file_size_bytes: File size in bytes (from Dropbox metadata), 0 if unknown

    Returns:
        True if processing succeeded, False otherwise.
    """
    watcher_config = config.dropbox_watcher
    rclone_remote = watcher_config.rclone_remote

    # Validate path for safety
    if not validate_dropbox_path(dropbox_path):
        logger.error(f"Invalid Dropbox path rejected: {dropbox_path!r}")
        return False

    # Check file size limit
    if watcher_config.max_file_size_mb > 0 and file_size_bytes > 0:
        max_bytes = watcher_config.max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.warning(
                f"File too large, skipping: {dropbox_path} "
                f"({file_size_mb:.1f} MB > {watcher_config.max_file_size_mb} MB limit)"
            )
            return False

    filename = os.path.basename(dropbox_path)
    base_name = os.path.splitext(filename)[0]

    # Create a unique temp directory
    request_id = uuid.uuid4().hex[:12]
    temp_dir = tempfile.mkdtemp(prefix=f"dropbox_{request_id}_")

    try:
        # Step 0: Send start notification
        if watcher_config.telegram_chat_id:
            try:
                escaped_path = html.escape(dropbox_path)
                file_size_str = ""
                if file_size_bytes > 0:
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    file_size_str = f" ({file_size_mb:.1f} MB)"
                await bot.send_message(
                    chat_id=watcher_config.telegram_chat_id,
                    text=f"🎙 Processing new audio file{file_size_str}:\n<code>{escaped_path}</code>",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning(f"Failed to send start notification: {e}")

        # Step 1: Download audio from Dropbox
        logger.info(f"Downloading audio: {dropbox_path}")
        local_audio_dir = os.path.join(temp_dir, "input")
        os.makedirs(local_audio_dir, exist_ok=True)

        src_path = f"{rclone_remote}:{dropbox_path}"
        if not await rclone_copy(src_path, local_audio_dir):
            raise RuntimeError(f"Failed to download audio: {dropbox_path}")

        local_audio_path = os.path.join(local_audio_dir, filename)
        if not os.path.exists(local_audio_path):
            raise RuntimeError(f"Downloaded file not found: {local_audio_path}")

        logger.info(f"Downloaded to: {local_audio_path}")

        # Step 2: Transcribe (with optional background context from CONTEXT.md)
        logger.info("Starting transcription...")

        # Create metadata from background if available
        transcription_metadata = None
        if context.background:
            transcription_metadata = TranscriptionMetadata(
                source_name="Dropbox Recording",
                title=base_name,
                description=context.background,
            )
            logger.info("Using background context from CONTEXT.md for transcription")

        raw_transcript = await transcribe(
            local_audio_path,
            config.transcriber,
            metadata=transcription_metadata,
            on_status=lambda s: logger.info(f"Transcribe: {s}"),
        )
        logger.info(f"Transcription complete, length: {len(raw_transcript)}")

        # Step 3: Edit/format (with optional custom sections from CONTEXT.md)
        logger.info("Starting editing...")
        sections_override = context.sections if context.sections else None
        edited_transcript = await edit(
            raw_transcript,
            config.editor,
            sections_override=sections_override,
            background=context.background,
            on_status=lambda s: logger.info(f"Edit: {s}"),
        )
        logger.info(f"Editing complete, length: {len(edited_transcript)}")

        # Step 4: Generate output files
        title = extract_title_from_transcript(edited_transcript)
        if title:
            safe_title = sanitize_filename(title, max_length=200)
        else:
            safe_title = sanitize_filename(base_name, max_length=200)

        date_stamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{safe_title} - {date_stamp}"

        # Save Markdown
        md_path = os.path.join(temp_dir, f"{output_filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(edited_transcript)

        # Generate PDF
        pdf_path = os.path.join(temp_dir, f"{output_filename}.pdf")
        await generate_pdf(edited_transcript, pdf_path)
        logger.info(f"Generated output files: {output_filename}")

        # Step 5: Upload results to Dropbox output folder
        # Track whether we have at least one output destination
        has_output_destination = bool(watcher_config.output_folder or watcher_config.telegram_chat_id)
        if not has_output_destination:
            logger.warning(
                f"No output destination configured (no output_folder or telegram_chat_id). "
                f"Processed files will remain in temp directory until cleanup."
            )

        upload_succeeded = False  # Default to False, only set True when upload succeeds
        if watcher_config.output_folder:
            output_folder = watcher_config.output_folder.rstrip("/")

            # Upload markdown
            md_dst = f"{rclone_remote}:{output_folder}/{output_filename}.md"
            md_uploaded = await rclone_copyto(md_path, md_dst)
            if not md_uploaded:
                logger.error(f"Failed to upload markdown to: {md_dst}")

            # Upload PDF
            pdf_dst = f"{rclone_remote}:{output_folder}/{output_filename}.pdf"
            pdf_uploaded = await rclone_copyto(pdf_path, pdf_dst)
            if not pdf_uploaded:
                logger.error(f"Failed to upload PDF to: {pdf_dst}")

            upload_succeeded = md_uploaded and pdf_uploaded
            if upload_succeeded:
                logger.info(f"Uploaded results to: {output_folder}")

        # Step 6: Send Telegram notification
        telegram_sent = False
        if watcher_config.telegram_chat_id:
            try:
                pdf_file = FSInputFile(pdf_path, filename=f"{output_filename}.pdf")
                # Use HTML mode and escape path to avoid Markdown injection issues
                escaped_path = html.escape(dropbox_path)
                await bot.send_document(
                    chat_id=watcher_config.telegram_chat_id,
                    document=pdf_file,
                    caption=f"✅ Transcription complete:\n<code>{escaped_path}</code>",
                    parse_mode="HTML",
                )
                logger.info(f"Sent notification to chat: {watcher_config.telegram_chat_id}")
                telegram_sent = True
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

        # Step 7: Move original audio to _processed subfolder
        # Only move if at least one output destination succeeded (Dropbox upload or Telegram)
        output_delivered = upload_succeeded or telegram_sent
        if not output_delivered:
            logger.warning(f"Skipping move of original file (no output delivered): {dropbox_path}")
        else:
            parent_dir = os.path.dirname(dropbox_path)
            processed_dir = f"{parent_dir}/{watcher_config.processed_subfolder}"
            processed_path = f"{processed_dir}/{filename}"

            if await rclone_moveto(f"{rclone_remote}:{dropbox_path}", f"{rclone_remote}:{processed_path}"):
                logger.info(f"Moved original to: {processed_path}")
            else:
                logger.error(f"Failed to move original file to: {processed_path}")

        logger.info(f"Successfully processed: {dropbox_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {dropbox_path}: {e}", exc_info=True)

        # Try to send error notification
        if watcher_config.telegram_chat_id:
            try:
                # Use HTML format to avoid Markdown injection issues
                escaped_path = html.escape(dropbox_path)
                escaped_error = html.escape(str(e))
                await bot.send_message(
                    chat_id=watcher_config.telegram_chat_id,
                    text=f"❌ Failed to process:\n<code>{escaped_path}</code>\n\nError: {escaped_error}",
                    parse_mode="HTML",
                )
            except Exception:
                pass

        return False

    finally:
        # Step 8: Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
