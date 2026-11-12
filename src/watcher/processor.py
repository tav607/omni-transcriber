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
from datetime import date
from pathlib import PurePosixPath
from typing import Callable

import yaml

from aiogram import Bot
from aiogram.types import FSInputFile

from ..config import AppConfig, DropboxWatcherConfig
from ..services.transcriber import Labeling, TranscriptionMetadata, recording_date, transcribe
from ..services.editor import RecordingInfo, edit
from ..services.pdf_generator import generate_pdf
from .context_loader import ContextConfig

logger = logging.getLogger(__name__)

# Pattern to extract h1 title from markdown
H1_TITLE_PATTERN = re.compile(r"^#\s+(.+?)$", re.MULTILINE)

# Supported audio extensions
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".wav", ".webm", ".ogg", ".opus", ".flac", ".aac"}

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


async def rclone_cat(path: str) -> str | None:
    """Text of a remote file, or None when rclone could not fetch it (the
    caller has already established that the file exists)."""
    proc = await asyncio.create_subprocess_exec(
        "rclone", "cat", path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(f"rclone cat failed for {path}: {stderr.decode(errors='replace').strip()}")
        return None
    try:
        return stdout.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(f"rclone cat: {path} is not UTF-8 text")
        return None


async def rclone_lsf(path: str) -> list[str] | None:
    """File names in a remote folder; [] when the folder does not exist yet,
    None when the listing failed for another reason."""
    proc = await asyncio.create_subprocess_exec(
        "rclone", "lsf", "--files-only", path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        return [line for line in stdout.decode("utf-8", errors="replace").splitlines() if line]
    if proc.returncode == 3:  # rclone: directory not found
        return []
    logger.warning(f"rclone lsf failed for {path}: {stderr.decode(errors='replace').strip()}")
    return None


def unique_name(base: str, taken: set[str], ext: str = ".md") -> str:
    """base, or "base (2)", "base (3)", ... : the first whose file is not taken.
    Two calls on one day can get the same title, and the upload would
    overwrite the earlier note."""
    if f"{base}{ext}" not in taken:
        return base
    n = 2
    while f"{base} ({n}){ext}" in taken:
        n += 1
    return f"{base} ({n})"


def note_front_matter(source: str, recorded: str, mode: str) -> str:
    """YAML front matter for the uploaded note: which audio file it came from,
    when that was recorded, and the mode it ran in."""
    # A date object, so YAML writes it bare and Obsidian reads a date property.
    data = {"source": source, "recorded": date.fromisoformat(recorded), "mode": mode}
    return "---\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False) + "---\n\n"


async def move_aside(dropbox_path: str, subfolder: str, to_rclone_path: Callable[[str], str]) -> bool:
    """Move an audio file into a sibling subfolder (_processed, _failed)."""
    parent_dir = os.path.dirname(dropbox_path)
    target = f"{parent_dir}/{subfolder}/{os.path.basename(dropbox_path)}"
    if await rclone_moveto(to_rclone_path(dropbox_path), to_rclone_path(target)):
        logger.info(f"Moved original to: {target}")
        return True
    logger.error(f"Failed to move original file to: {target}")
    return False


async def process_audio_file(
    dropbox_path: str,
    context: ContextConfig,
    bot: Bot,
    config: AppConfig,
    file_size_bytes: int = 0,
    client_modified: date | None = None,
) -> bool:
    """
    Process an audio file from Dropbox.

    1. Download audio from Dropbox using rclone
    2. Transcribe using Gemini, labeled as the folder's CONTEXT.md says
    3. Edit/format using Gemini, in the CONTEXT.md mode and sections
    4. Generate PDF
    5. Upload the note to the Dropbox output folder
    6. Send Telegram notification
    7. Move the original audio to the _processed subfolder
    8. Clean up local temp files

    A file that fails is moved to the _failed subfolder, so a restart's scan
    does not run it again; moving it back out retries it.

    Args:
        dropbox_path: Dropbox path to the audio file, e.g., "/Recordings/file.m4a"
        context: settings in force for the file (see context_loader)
        bot: Telegram bot instance for notifications
        config: Application config
        file_size_bytes: File size in bytes (from Dropbox metadata), 0 if unknown
        client_modified: the file's own modification date as Dropbox reports
            it, the fallback recording date when the file name carries none

    Returns:
        True if processing succeeded, False otherwise.
    """
    watcher_config = config.dropbox_watcher
    rclone_remote = watcher_config.rclone_remote
    rclone_base_path = watcher_config.rclone_base_path.rstrip("/")

    def to_rclone_path(sdk_path: str) -> str:
        """Convert Dropbox SDK path to rclone full path."""
        if rclone_base_path:
            return f"{rclone_remote}:{rclone_base_path}{sdk_path}"
        return f"{rclone_remote}:{sdk_path}"

    # Validate path for safety
    if not validate_dropbox_path(dropbox_path):
        logger.error(f"Invalid Dropbox path rejected: {dropbox_path!r}")
        return False

    filename = os.path.basename(dropbox_path)
    base_name = os.path.splitext(filename)[0]

    # Create a unique temp directory
    request_id = uuid.uuid4().hex[:12]
    temp_dir = tempfile.mkdtemp(prefix=f"dropbox_{request_id}_")

    # Status message for progress updates
    status_message = None
    # Remove leading "/" to avoid Telegram treating it as a command
    display_path = dropbox_path.lstrip("/")
    escaped_path = html.escape(display_path)
    settings_line = f"<i>{html.escape(context.describe())}</i>"

    async def update_status(text: str):
        """Update the status message with new text."""
        nonlocal status_message
        if not watcher_config.telegram_chat_id:
            return
        try:
            if status_message:
                await status_message.edit_text(text, parse_mode="HTML")
            else:
                status_message = await bot.send_message(
                    chat_id=watcher_config.telegram_chat_id,
                    text=text,
                    parse_mode="HTML",
                )
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")

    try:
        # Check file size limit
        if watcher_config.max_file_size_mb > 0 and file_size_bytes > 0:
            max_bytes = watcher_config.max_file_size_mb * 1024 * 1024
            if file_size_bytes > max_bytes:
                raise RuntimeError(
                    f"File is {file_size_bytes / (1024 * 1024):.1f} MB, over the "
                    f"{watcher_config.max_file_size_mb} MB limit"
                )

        # Step 0: Send start notification, with the settings the run uses
        file_size_str = ""
        if file_size_bytes > 0:
            file_size_mb = file_size_bytes / (1024 * 1024)
            file_size_str = f" ({file_size_mb:.1f} MB)"
        await update_status(
            f"🎙 Processing new audio file{file_size_str}:\n<code>{escaped_path}</code>\n{settings_line}"
        )

        # Step 1: Download audio from Dropbox
        await update_status(
            f"⏬ Downloading audio{file_size_str}...\n<code>{escaped_path}</code>\n{settings_line}"
        )
        logger.info(f"Downloading audio: {dropbox_path}")
        local_audio_dir = os.path.join(temp_dir, "input")
        os.makedirs(local_audio_dir, exist_ok=True)

        src_path = to_rclone_path(dropbox_path)
        if not await rclone_copy(src_path, local_audio_dir):
            raise RuntimeError(f"Failed to download audio: {dropbox_path}")

        local_audio_path = os.path.join(local_audio_dir, filename)
        if not os.path.exists(local_audio_path):
            raise RuntimeError(f"Downloaded file not found: {local_audio_path}")

        logger.info(f"Downloaded to: {local_audio_path}")

        # Step 2: Transcribe
        await update_status(f"🎙 Transcribing audio...\n<code>{escaped_path}</code>\n{settings_line}")
        logger.info("Starting transcription...")

        # The date anchors spoken years (see recording_date): a date in the
        # file name, otherwise the file's own modification date as the latest
        # the recording can be.
        recorded, date_is_upper_bound = recording_date(filename, client_modified)
        transcription_metadata = TranscriptionMetadata(
            source_name="Dropbox Recording",
            title=base_name,
            publish_date=recorded,
            description=context.background,
            date_is_upper_bound=date_is_upper_bound,
        )

        raw_transcript = await transcribe(
            local_audio_path,
            config.transcriber,
            metadata=transcription_metadata,
            on_status=lambda s: logger.info(f"Transcribe: {s}"),
            labeling=Labeling(
                mode=context.mode, roles=context.roles, participants=context.participants
            ),
        )
        logger.info(f"Transcription complete, length: {len(raw_transcript)}")

        # Step 3: Edit/format
        await update_status(f"✏️ Formatting transcript...\n<code>{escaped_path}</code>\n{settings_line}")
        logger.info("Starting editing...")
        edited_transcript = await edit(
            raw_transcript,
            config.editor,
            sections=context.sections,
            enable_translation=context.translation,
            background=context.background,
            on_status=lambda s: logger.info(f"Edit: {s}"),
            mode=context.mode,
            roles=context.roles,
            glossary=context.glossary,
            recording=RecordingInfo(base_name, recorded, date_is_upper_bound),
        )
        del raw_transcript  # Free memory early
        logger.info(f"Editing complete, length: {len(edited_transcript)}")

        # Step 4: Generate output files, named by title and recording date
        await update_status(f"📄 Generating output files...\n<code>{escaped_path}</code>")
        title = extract_title_from_transcript(edited_transcript)
        safe_title = sanitize_filename(title or base_name, max_length=200)
        output_filename = f"{safe_title} - {recorded.replace('-', '')}"

        output_folder = watcher_config.output_folder.rstrip("/")
        if output_folder and context.output_folder:
            output_folder = f"{output_folder}/{context.output_folder}"
        if output_folder:
            # Never write over an existing note of the same name.
            taken = await rclone_lsf(f"{rclone_remote}:{output_folder}")
            if taken is not None:
                output_filename = unique_name(output_filename, set(taken))

        # The uploaded note carries front matter; the PDF does not.
        md_path = os.path.join(temp_dir, f"{output_filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(note_front_matter(display_path, recorded, context.mode) + edited_transcript)

        # Generate PDF (degrade to Markdown-only if the PDF toolchain fails; the
        # Markdown uploaded to Dropbox below is the watcher's primary deliverable)
        pdf_path = os.path.join(temp_dir, f"{output_filename}.pdf")
        pdf_generated = True
        try:
            await generate_pdf(edited_transcript, pdf_path)
            logger.info(f"Generated output files: {output_filename}")
        except Exception as e:
            pdf_generated = False
            logger.error(f"PDF generation failed, delivering Markdown only: {e}")

        # Step 5: Upload results to Dropbox output folder
        # Track whether we have at least one output destination
        has_output_destination = bool(output_folder or watcher_config.telegram_chat_id)
        if not has_output_destination:
            logger.warning(
                "No output destination configured (no output_folder or telegram_chat_id). "
                "Processed files will remain in temp directory until cleanup."
            )

        upload_succeeded = False  # Default to False, only set True when upload succeeds
        if output_folder:
            await update_status(f"📤 Uploading to Dropbox...\n<code>{escaped_path}</code>")

            # Upload markdown only (PDF is sent via Telegram)
            md_dst = f"{rclone_remote}:{output_folder}/{output_filename}.md"
            upload_succeeded = await rclone_copyto(md_path, md_dst)
            if not upload_succeeded:
                logger.error(f"Failed to upload markdown to: {md_dst}")
            if upload_succeeded:
                logger.info(f"Uploaded results to: {output_folder}")

        # Step 6: Send Telegram notification
        telegram_sent = False
        if watcher_config.telegram_chat_id:
            try:
                await update_status(f"📤 Sending files...\n<code>{escaped_path}</code>")
                if pdf_generated:
                    doc_file = FSInputFile(pdf_path, filename=f"{output_filename}.pdf")
                    caption = f"✅ Transcription complete:\n{escaped_path}"
                else:
                    doc_file = FSInputFile(md_path, filename=f"{output_filename}.md")
                    caption = f"⚠️ Transcription complete (PDF failed, sent Markdown):\n{escaped_path}"
                await bot.send_document(
                    chat_id=watcher_config.telegram_chat_id,
                    document=doc_file,
                    caption=caption,
                    parse_mode="HTML",
                )
                # Delete status message after sending the document
                if status_message:
                    try:
                        await status_message.delete()
                    except Exception:
                        pass  # Ignore deletion errors
                logger.info(f"Sent notification to chat: {watcher_config.telegram_chat_id}")
                telegram_sent = True
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

        # Step 7: Move original audio to _processed subfolder
        # Only move if at least one output destination succeeded (Dropbox upload or Telegram)
        if not (upload_succeeded or telegram_sent):
            raise RuntimeError("The transcript was produced but could not be delivered anywhere")
        await move_aside(dropbox_path, watcher_config.processed_subfolder, to_rclone_path)

        logger.info(f"Successfully processed: {dropbox_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {dropbox_path}: {e}", exc_info=True)

        # Out of the watched folder, so the next startup scan does not run it
        # again; the message says where it went.
        moved = await move_aside(dropbox_path, watcher_config.failed_subfolder, to_rclone_path)
        where = (
            f"Moved to <code>{html.escape(watcher_config.failed_subfolder)}/</code>; "
            "move it back out to retry."
            if moved else "It is still in its folder and will be tried again at the next restart."
        )
        escaped_error = html.escape(str(e))
        await update_status(
            f"❌ Failed to process:\n<code>{escaped_path}</code>\n\nError: {escaped_error}\n\n{where}"
        )

        return False

    finally:
        # Step 8: Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
