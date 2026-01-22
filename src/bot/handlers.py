import asyncio
import logging
import os
import re
import shutil
import stat
import subprocess
import uuid
from dataclasses import replace
from datetime import datetime

from aiogram import Router, F
from aiogram.types import Message, FSInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters import Command

from ..config import config
from ..utils.url_parser import is_youtube_url, is_bilibili_url, is_apple_podcasts_url, is_xiaoyuzhou_url, is_supported_url, get_url_platform, extract_video_id, extract_xiaoyuzhou_episode_id
from ..utils import settings_store
from ..services.downloader import download_audio, download_audio_from_url, DownloadResult, VideoMetadata
from ..services.transcriber import transcribe, TranscriptionMetadata
from ..services.editor import edit, edit_podcast, PodcastEpisodeMetadata, VideoEditorMetadata
from ..services.rss_parser import get_episode_metadata
from ..services.xiaoyuzhou_parser import get_episode_metadata as get_xiaoyuzhou_metadata
from ..services.pdf_generator import generate_pdf

logger = logging.getLogger(__name__)
router = Router()

# Initialize settings store on module load
settings_store.init()

# Model options
MODELS = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3-pro-preview",
}

# Default models
DEFAULT_TRANSCRIBER_MODEL = "flash"
DEFAULT_EDITOR_MODEL = "pro"

# Supported audio MIME types
AUDIO_MIME_TYPES = [
    "audio/mpeg",  # mp3
    "audio/mp4",  # m4a
    "audio/x-m4a",  # m4a alternative
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
]

# Pattern to extract h1 title from markdown
H1_TITLE_PATTERN = re.compile(r"^#\s+(.+?)$", re.MULTILINE)


def extract_title_from_transcript(transcript: str) -> str | None:
    """
    Extract the h1 title from a markdown transcript.

    Returns the title text or None if not found.
    """
    match = H1_TITLE_PATTERN.search(transcript)
    if match:
        return match.group(1).strip()
    return None


def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """
    Sanitize a filename by removing illegal characters.

    - Extracts basename to remove any path components
    - Removes illegal filesystem characters
    - Collapses multiple spaces into one
    - Limits length to prevent filesystem issues
    """
    # Extract just the filename, removing any path components
    filename = os.path.basename(filename)

    # Split into name and extension
    name, ext = os.path.splitext(filename)

    # Remove illegal characters (keep spaces, Chinese chars, alphanumeric, hyphen, dot)
    illegal_chars = r'[\\:*?"<>|\x00-\x1f]'
    name = re.sub(illegal_chars, '', name)
    ext = re.sub(illegal_chars, '', ext)

    # Collapse multiple spaces into one
    name = re.sub(r'\s+', ' ', name).strip()

    # Limit length
    if len(name) > max_length:
        name = name[:max_length].strip()

    # Ensure we have a valid name
    if not name:
        name = "file"

    return name + ext


async def upload_to_rclone(file_path: str, filename: str, chat_id: int) -> bool:
    """
    Upload a file to rclone destination if enabled for this chat.

    Args:
        file_path: Path to the file to upload
        filename: Desired filename in destination
        chat_id: Chat ID of the user

    Returns:
        True if upload succeeded, False otherwise
    """
    # Check if rclone is enabled and user is allowed
    if not config.rclone.is_enabled:
        return False

    if chat_id not in config.rclone.enabled_chat_ids:
        return False

    # Construct full rclone path: remote:path/filename
    upload_path = config.rclone.upload_path.rstrip("/")
    destination = f"{config.rclone.remote}:{upload_path}/{filename}"
    logger.info(f"Uploading to rclone: {destination}")

    try:
        # Run rclone copy in a subprocess
        process = await asyncio.create_subprocess_exec(
            "rclone", "copyto", file_path, destination,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Rclone upload failed: {stderr.decode()}")
            return False

        logger.info(f"Rclone upload succeeded: {destination}")
        return True

    except Exception as e:
        logger.error(f"Rclone upload error: {e}")
        return False


@router.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(
        "Welcome to the AI Transcriber Bot!\n\n"
        "I can help you transcribe audio from:\n"
        "- YouTube videos\n"
        "- Bilibili videos\n"
        "- Apple Podcasts\n"
        "- Xiaoyuzhou Podcasts (小宇宙)\n"
        "- Audio files\n\n"
        "Just send me a URL or audio file!\n\n"
        "I'll generate a formatted transcript with summary and key points, "
        "delivered as both Markdown and PDF files."
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    await message.answer(
        "*How to use this bot:*\n\n"
        "*Supported URLs:*\n"
        "• YouTube: `youtube.com/watch?v=...`\n"
        "• Bilibili: `bilibili.com/video/BV...`\n"
        "• Apple Podcasts: `podcasts.apple.com/...`\n"
        "• Xiaoyuzhou: `xiaoyuzhoufm.com/episode/...`\n\n"
        "*Audio Files:*\n"
        "Send me an audio file (mp3, m4a, wav, webm, etc.)\n\n"
        "*Settings:*\n"
        "`/model` - Choose AI model (Flash/Pro)\n"
        "`/translation` - Toggle inline Chinese translation\n\n"
        "*Output:*\n"
        "You'll receive:\n"
        "- A Markdown file with the formatted transcript\n"
        "- A PDF file for easy reading and sharing",
        parse_mode="Markdown",
    )


@router.message(Command("translation"))
async def cmd_translation(message: Message):
    """Handle /translation command - show translation mode selection menu."""
    chat_id = message.chat.id

    # Get current setting
    current_value = settings_store.get(chat_id, "translation", False)

    # Build inline keyboard
    keyboard = [
        [
            InlineKeyboardButton(
                text=f"{'✓ ' if not current_value else ''}Off",
                callback_data="translation_off",
            ),
            InlineKeyboardButton(
                text=f"{'✓ ' if current_value else ''}On",
                callback_data="translation_on",
            ),
        ],
    ]

    status = "ON" if current_value else "OFF"
    await message.answer(
        f"*Translation Settings*\n\n"
        f"Current: *{status}*\n\n"
        "When enabled, non-Chinese transcripts will include inline Chinese translations:\n"
        "`Original text here.`\n"
        "`> 中文翻译在这里。`",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
        parse_mode="Markdown",
    )


@router.callback_query(F.data.startswith("translation_"))
async def translation_callback(callback: CallbackQuery):
    """Handle translation selection callbacks."""
    if not callback.data or not callback.message:
        return

    await callback.answer()

    chat_id = callback.message.chat.id
    new_value = callback.data == "translation_on"

    # Get current value
    current_value = settings_store.get(chat_id, "translation", False)

    if new_value == current_value:
        # Already selected, just delete the menu
        try:
            await callback.message.delete()
        except Exception:
            pass
        return

    # Update setting
    settings_store.set(chat_id, "translation", new_value)

    # Delete the menu and send confirmation
    try:
        await callback.message.delete()
    except Exception:
        pass

    status = "ON" if new_value else "OFF"
    await callback.message.answer(
        f"✓ Translation mode set to *{status}*",
        parse_mode="Markdown",
    )


@router.message(Command("model"))
async def cmd_model(message: Message):
    """Handle /model command - show model selection menu."""
    chat_id = message.chat.id

    # Get current models
    current_transcriber = settings_store.get(chat_id, "transcriber_model", DEFAULT_TRANSCRIBER_MODEL)
    current_editor = settings_store.get(chat_id, "editor_model", DEFAULT_EDITOR_MODEL)

    # Build inline keyboard
    keyboard = [
        [InlineKeyboardButton(
            text="── Transcriber ──",
            callback_data="model_noop",
        )],
        [
            InlineKeyboardButton(
                text=f"{'✓ ' if current_transcriber == 'flash' else ''}Flash",
                callback_data="model_transcriber_flash",
            ),
            InlineKeyboardButton(
                text=f"{'✓ ' if current_transcriber == 'pro' else ''}Pro",
                callback_data="model_transcriber_pro",
            ),
        ],
        [InlineKeyboardButton(
            text="── Editor ──",
            callback_data="model_noop",
        )],
        [
            InlineKeyboardButton(
                text=f"{'✓ ' if current_editor == 'flash' else ''}Flash",
                callback_data="model_editor_flash",
            ),
            InlineKeyboardButton(
                text=f"{'✓ ' if current_editor == 'pro' else ''}Pro",
                callback_data="model_editor_pro",
            ),
        ],
    ]

    await message.answer(
        "*Model Settings*\n\n"
        f"Transcriber: *{current_transcriber.upper()}* (`{MODELS[current_transcriber]}`)\n"
        f"Editor: *{current_editor.upper()}* (`{MODELS[current_editor]}`)\n\n"
        "Select model for each component:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
        parse_mode="Markdown",
    )


@router.callback_query(F.data.startswith("model_"))
async def model_callback(callback: CallbackQuery):
    """Handle model selection callbacks."""
    if not callback.data or not callback.message:
        return

    await callback.answer()

    # Ignore noop callbacks (section headers)
    if callback.data == "model_noop":
        return

    chat_id = callback.message.chat.id
    data = callback.data

    # Parse callback data: model_<component>_<model>
    parts = data.split("_")
    if len(parts) != 3:
        return

    _, component, model = parts
    if component not in ("transcriber", "editor") or model not in ("flash", "pro"):
        return

    # Update settings
    settings_key = f"{component}_model"
    default_model = DEFAULT_TRANSCRIBER_MODEL if component == "transcriber" else DEFAULT_EDITOR_MODEL
    old_model = settings_store.get(chat_id, settings_key, default_model)

    if model == old_model:
        # Already selected, just delete the menu
        try:
            await callback.message.delete()
        except Exception:
            pass
        await callback.message.chat.do(
            action="typing"
        )  # Dummy action to avoid "query is too old" error
        return

    settings_store.set(chat_id, settings_key, model)

    # Delete the menu and send confirmation
    try:
        await callback.message.delete()
    except Exception:
        pass

    component_name = "Transcriber" if component == "transcriber" else "Editor"
    await callback.message.answer(
        f"✓ {component_name} model set to *{model.upper()}*\n"
        f"(`{MODELS[model]}`)",
        parse_mode="Markdown",
    )


@router.message(F.audio | F.voice | F.document)
async def handle_audio(message: Message):
    """Handle audio file uploads."""
    # Get the file object
    if message.audio:
        file = message.audio
        file_name = file.file_name or f"audio_{uuid.uuid4().hex[:8]}.mp3"
    elif message.voice:
        file = message.voice
        file_name = f"voice_{uuid.uuid4().hex[:8]}.ogg"
    elif message.document:
        file = message.document
        # Check MIME type - only accept audio files
        if file.mime_type:
            # Allow webm files even if marked as video/webm (often contains audio only)
            is_webm = file.mime_type == "video/webm" or (
                file.file_name and file.file_name.lower().endswith(".webm")
            )
            if file.mime_type.startswith("video/") and not is_webm:
                # Reject video files - they need ffmpeg extraction which we don't support
                await message.answer(
                    "Video files are not supported. "
                    "Please extract the audio first or send an audio file directly."
                )
                return
            if not file.mime_type.startswith("audio/") and not is_webm:
                # Not an audio file, ignore
                return
        file_name = file.file_name or f"audio_{uuid.uuid4().hex[:8]}"
    else:
        return

    logger.info(f"Received audio file: {file_name}")
    status_message = await message.answer("Received audio file. Processing...")

    # Get caption for metadata context
    caption = message.caption

    try:
        await _process_audio_file(message, file, file_name, status_message, caption=caption)
    except Exception as e:
        logger.error(f"Error processing audio file: {e}", exc_info=True)
        await status_message.edit_text(f"Error processing audio file: {str(e)}")


@router.message(F.text)
async def handle_text(message: Message):
    """Handle text messages (check for YouTube/Bilibili/Apple Podcasts URLs)."""
    text = message.text
    if not text:
        return

    # Check if it's a supported video URL
    platform = get_url_platform(text)

    if platform == "youtube":
        video_id = extract_video_id(text)
        logger.info(f"Received YouTube URL, video_id: {video_id}")
        status_message = await message.answer(
            f"Detected YouTube video. Processing...\nVideo ID: `{video_id}`",
            parse_mode="Markdown",
        )

        try:
            await _process_video_url(message, text, status_message, platform)
        except Exception as e:
            logger.error(f"Error processing YouTube URL: {e}", exc_info=True)
            await status_message.edit_text(f"Error processing YouTube video: {str(e)}")

    elif platform == "bilibili":
        logger.info(f"Received Bilibili URL: {text}")
        status_message = await message.answer(
            "Detected Bilibili video. Processing...",
            parse_mode="Markdown",
        )

        try:
            await _process_video_url(message, text, status_message, platform)
        except Exception as e:
            logger.error(f"Error processing Bilibili URL: {e}", exc_info=True)
            await status_message.edit_text(f"Error processing Bilibili video: {str(e)}")

    elif platform == "apple_podcasts":
        logger.info(f"Received Apple Podcasts URL: {text}")
        status_message = await message.answer(
            "Detected Apple Podcasts. Processing...",
            parse_mode="Markdown",
        )

        try:
            await _process_video_url(message, text, status_message, platform)
        except Exception as e:
            logger.error(f"Error processing Apple Podcasts URL: {e}", exc_info=True)
            await status_message.edit_text(f"Error processing podcast: {str(e)}")

    elif platform == "xiaoyuzhou":
        logger.info(f"Received Xiaoyuzhou URL: {text}")
        status_message = await message.answer(
            "Detected Xiaoyuzhou podcast. Processing...",
            parse_mode="Markdown",
        )

        try:
            await _process_video_url(message, text, status_message, platform)
        except Exception as e:
            logger.error(f"Error processing Xiaoyuzhou URL: {e}", exc_info=True)
            await status_message.edit_text(f"Error processing podcast: {str(e)}")

    else:
        # Not a supported URL, ignore or send help
        await message.answer(
            "Please send me a URL (YouTube/Bilibili/Apple Podcasts/Xiaoyuzhou) or an audio file.\n"
            "Use /help for more information."
        )


async def _process_video_url(
    message: Message, url: str, status_message: Message, platform: str
) -> None:
    """Process a video URL (YouTube/Bilibili/Apple Podcasts) and send the transcript."""
    # Get user settings
    chat_id = message.chat.id
    enable_translation = settings_store.get(chat_id, "translation", False)

    # Get user model preferences
    transcriber_model_key = settings_store.get(chat_id, "transcriber_model", DEFAULT_TRANSCRIBER_MODEL)
    editor_model_key = settings_store.get(chat_id, "editor_model", DEFAULT_EDITOR_MODEL)
    transcriber_model = MODELS[transcriber_model_key]
    editor_model = MODELS[editor_model_key]

    # Create config overrides if user selected different models
    transcriber_config = config.transcriber
    if transcriber_model != config.transcriber.model:
        transcriber_config = replace(config.transcriber, model=transcriber_model)

    editor_config = config.editor
    if editor_model != config.editor.model:
        editor_config = replace(config.editor, model=editor_model)

    # Create a unique temporary directory for this request to prevent collisions
    request_id = uuid.uuid4().hex[:12]
    platform_prefixes = {"youtube": "yt", "bilibili": "bili", "apple_podcasts": "pod", "xiaoyuzhou": "xyz"}
    platform_prefix = platform_prefixes.get(platform, "media")
    request_temp_dir = os.path.join(config.temp_dir, f"{platform_prefix}_{request_id}")
    os.makedirs(request_temp_dir, exist_ok=True)

    platform_names = {"youtube": "YouTube", "bilibili": "Bilibili", "apple_podcasts": "Apple Podcasts", "xiaoyuzhou": "Xiaoyuzhou"}
    platform_name = platform_names.get(platform, "source")

    try:
        # ============================================
        # Apple Podcasts: Use specialized podcast mode
        # ============================================
        if platform == "apple_podcasts":
            await _process_apple_podcast(
                message=message,
                url=url,
                status_message=status_message,
                transcriber_config=transcriber_config,
                editor_config=editor_config,
                request_temp_dir=request_temp_dir,
                chat_id=chat_id,
            )
            return

        # ============================================
        # Xiaoyuzhou: Use specialized podcast mode (via RSSHub)
        # ============================================
        if platform == "xiaoyuzhou":
            await _process_xiaoyuzhou_podcast(
                message=message,
                url=url,
                status_message=status_message,
                transcriber_config=transcriber_config,
                editor_config=editor_config,
                request_temp_dir=request_temp_dir,
                chat_id=chat_id,
            )
            return

        # ============================================
        # YouTube/Bilibili: Use standard processing
        # ============================================

        # Download audio
        await status_message.edit_text(f"Downloading audio from {platform_name}...")
        download_result = await download_audio(url, request_temp_dir)
        audio_path = download_result.audio_path
        video_metadata = download_result.metadata

        if video_metadata:
            logger.info(f"Got video metadata: {video_metadata.title} by {video_metadata.channel}")

        # Convert VideoMetadata to TranscriptionMetadata for transcriber
        transcription_metadata = None
        if video_metadata:
            transcription_metadata = TranscriptionMetadata(
                source_name=video_metadata.channel,
                title=video_metadata.title,
                publish_date=video_metadata.upload_date,
                description=video_metadata.description,
            )

        # Transcribe
        await status_message.edit_text("Transcribing audio...")
        raw_transcript = await transcribe(
            audio_path,
            transcriber_config,
            metadata=transcription_metadata,
            on_status=lambda s: logger.info(s),
        )

        # Convert VideoMetadata to VideoEditorMetadata for editor
        editor_metadata = None
        if video_metadata:
            editor_metadata = VideoEditorMetadata(
                title=video_metadata.title,
                channel=video_metadata.channel,
                description=video_metadata.description,
                upload_date=video_metadata.upload_date,
                webpage_url=video_metadata.webpage_url,
            )

        # Edit/format
        await status_message.edit_text("Formatting transcript...")
        edited_transcript = await edit(
            raw_transcript,
            editor_config,
            enable_translation=enable_translation,
            metadata=editor_metadata,
            on_status=lambda s: logger.info(s),
        )

        # Generate output files
        await status_message.edit_text("Generating output files...")

        # Use video title from metadata for filename, fallback to extracted title
        title = None
        if video_metadata and video_metadata.title:
            title = video_metadata.title
        else:
            title = extract_title_from_transcript(edited_transcript)

        if title:
            # Sanitize the title for use as filename
            safe_title = sanitize_filename(title, max_length=200)
        else:
            # Fallback to video_id/platform if no title found
            if platform == "youtube":
                safe_title = extract_video_id(url) or "transcript"
            else:
                safe_title = "transcript"

        # Add date stamp
        date_stamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{safe_title} - {date_stamp}"

        # Save Markdown
        md_path = os.path.join(request_temp_dir, "transcript.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(edited_transcript)

        # Generate PDF
        pdf_path = os.path.join(request_temp_dir, "transcript.pdf")
        await generate_pdf(edited_transcript, pdf_path)

        # Upload to rclone if enabled for this user
        rclone_uploaded = await upload_to_rclone(md_path, f"{output_filename}.md", chat_id)

        # Send files
        await status_message.edit_text("Sending files...")

        pdf_file = FSInputFile(pdf_path, filename=f"{output_filename}.pdf")

        # Only send .md if rclone upload was not successful
        if not rclone_uploaded:
            md_file = FSInputFile(md_path, filename=f"{output_filename}.md")
            await message.answer_document(md_file, caption="Markdown transcript")

        await message.answer_document(pdf_file, caption="PDF transcript")

        if rclone_uploaded:
            await status_message.edit_text("Done! Your transcript is ready. (Markdown synced to Dropbox)")
        else:
            await status_message.edit_text("Done! Your transcript is ready.")

    finally:
        # Cleanup entire temp directory for this request
        try:
            shutil.rmtree(request_temp_dir)
        except Exception:
            pass


async def _process_apple_podcast(
    message: Message,
    url: str,
    status_message: Message,
    transcriber_config,
    editor_config,
    request_temp_dir: str,
    chat_id: int,
) -> None:
    """
    Process an Apple Podcast URL using specialized podcast mode.

    This uses:
    - Episode metadata for context (podcast name, title, shownotes)
    - Podcast-specific transcription prompt
    - Podcast-specific editor with structured output (Info, Summary, Takeaways, Q&A, Highlights)
    """
    # Step 1: Get episode metadata from RSS feed
    await status_message.edit_text("Fetching podcast metadata...")
    episode_metadata = await get_episode_metadata(url)

    if episode_metadata:
        logger.info(f"Got podcast metadata: {episode_metadata.podcast_name} - {episode_metadata.episode_title}")
    else:
        logger.warning("Could not get podcast metadata, proceeding without context")

    # Step 2: Download audio
    await status_message.edit_text("Downloading podcast audio...")
    download_result = await download_audio(url, request_temp_dir)
    audio_path = download_result.audio_path

    # Step 3: Transcribe with metadata context
    await status_message.edit_text("Transcribing podcast...")

    # Convert to TranscriptionMetadata for transcriber
    transcription_metadata = None
    if episode_metadata:
        transcription_metadata = TranscriptionMetadata(
            source_name=episode_metadata.podcast_name,
            title=episode_metadata.episode_title,
            publish_date=episode_metadata.publish_date,
            description=episode_metadata.shownotes,
        )

    raw_transcript = await transcribe(
        audio_path,
        transcriber_config,
        metadata=transcription_metadata,
        on_status=lambda s: logger.info(s),
    )

    # Step 4: Edit with podcast mode (structured output)
    await status_message.edit_text("Formatting transcript (podcast mode)...")

    # Convert to PodcastEpisodeMetadata for editor
    editor_metadata = None
    if episode_metadata:
        editor_metadata = PodcastEpisodeMetadata(
            podcast_name=episode_metadata.podcast_name,
            episode_title=episode_metadata.episode_title,
            episode_link=episode_metadata.episode_link,
            publish_date=episode_metadata.publish_date,
            shownotes=episode_metadata.shownotes,
        )

    edited_result = await edit_podcast(
        raw_transcript,
        editor_config,
        metadata=editor_metadata,
        on_status=lambda s: logger.info(s),
    )

    # Step 5: Generate output files
    await status_message.edit_text("Generating output files...")

    # Use podcast title for filename
    if editor_metadata:
        safe_title = sanitize_filename(
            f"{editor_metadata.podcast_name} - {editor_metadata.episode_title}",
            max_length=200
        )
    else:
        safe_title = sanitize_filename(edited_result.title, max_length=200) or "podcast"

    # Add date stamp
    date_stamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"{safe_title} - {date_stamp}"

    # Save Markdown (use the structured markdown from podcast result)
    md_path = os.path.join(request_temp_dir, "transcript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(edited_result.markdown)

    # Generate PDF
    pdf_path = os.path.join(request_temp_dir, "transcript.pdf")
    await generate_pdf(edited_result.markdown, pdf_path)

    # Upload to rclone if enabled for this user
    rclone_uploaded = await upload_to_rclone(md_path, f"{output_filename}.md", chat_id)

    # Send files
    await status_message.edit_text("Sending files...")

    pdf_file = FSInputFile(pdf_path, filename=f"{output_filename}.pdf")

    # Only send .md if rclone upload was not successful
    if not rclone_uploaded:
        md_file = FSInputFile(md_path, filename=f"{output_filename}.md")
        await message.answer_document(md_file, caption="Markdown transcript")

    await message.answer_document(pdf_file, caption="PDF transcript")

    # Show completion message (unified with other sources)
    if rclone_uploaded:
        await status_message.edit_text("Done! Your transcript is ready. (Markdown synced to Dropbox)")
    else:
        await status_message.edit_text("Done! Your transcript is ready.")


async def _process_xiaoyuzhou_podcast(
    message: Message,
    url: str,
    status_message: Message,
    transcriber_config,
    editor_config,
    request_temp_dir: str,
    chat_id: int,
) -> None:
    """
    Process a Xiaoyuzhou podcast URL using specialized podcast mode via RSSHub.

    This uses:
    - RSSHub to fetch podcast RSS and extract episode metadata/audio URL
    - Episode metadata for context (podcast name, title, shownotes)
    - Podcast-specific transcription prompt
    - Podcast-specific editor with structured output (Info, Summary, Takeaways, Q&A, Highlights)
    """
    # Step 1: Extract episode ID from URL
    episode_id = extract_xiaoyuzhou_episode_id(url)
    if not episode_id:
        raise ValueError("Could not extract episode ID from URL")

    # Step 2: Get episode metadata via RSSHub
    await status_message.edit_text("Fetching podcast metadata from RSSHub...")
    episode_metadata = await get_xiaoyuzhou_metadata(episode_id)

    if not episode_metadata:
        raise ValueError(
            "Could not fetch episode metadata. "
            "Please ensure RSSHUB_BASE_URL is configured correctly."
        )

    if not episode_metadata.audio_url:
        raise ValueError("No audio URL found in episode metadata")

    logger.info(
        f"Got xiaoyuzhou metadata: {episode_metadata.podcast_name} - "
        f"{episode_metadata.episode_title}"
    )

    # Step 3: Download audio from direct URL
    await status_message.edit_text("Downloading podcast audio...")
    download_result = await download_audio_from_url(
        audio_url=episode_metadata.audio_url,
        output_dir=request_temp_dir,
        filename_prefix=f"xyz_{episode_id[:8]}",
    )
    audio_path = download_result.audio_path

    # Step 4: Transcribe with metadata context
    await status_message.edit_text("Transcribing podcast...")

    transcription_metadata = TranscriptionMetadata(
        source_name=episode_metadata.podcast_name,
        title=episode_metadata.episode_title,
        publish_date=episode_metadata.publish_date,
        description=episode_metadata.shownotes,
    )

    raw_transcript = await transcribe(
        audio_path,
        transcriber_config,
        metadata=transcription_metadata,
        on_status=lambda s: logger.info(s),
    )

    # Step 5: Edit with podcast mode (structured output)
    await status_message.edit_text("Formatting transcript (podcast mode)...")

    editor_metadata = PodcastEpisodeMetadata(
        podcast_name=episode_metadata.podcast_name,
        episode_title=episode_metadata.episode_title,
        episode_link=episode_metadata.episode_link,
        publish_date=episode_metadata.publish_date,
        shownotes=episode_metadata.shownotes,
    )

    edited_result = await edit_podcast(
        raw_transcript,
        editor_config,
        metadata=editor_metadata,
        on_status=lambda s: logger.info(s),
    )

    # Step 6: Generate output files
    await status_message.edit_text("Generating output files...")

    safe_title = sanitize_filename(
        f"{editor_metadata.podcast_name} - {editor_metadata.episode_title}",
        max_length=200
    )

    # Add date stamp
    date_stamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"{safe_title} - {date_stamp}"

    # Save Markdown
    md_path = os.path.join(request_temp_dir, "transcript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(edited_result.markdown)

    # Generate PDF
    pdf_path = os.path.join(request_temp_dir, "transcript.pdf")
    await generate_pdf(edited_result.markdown, pdf_path)

    # Upload to rclone if enabled for this user
    rclone_uploaded = await upload_to_rclone(md_path, f"{output_filename}.md", chat_id)

    # Send files
    await status_message.edit_text("Sending files...")

    pdf_file = FSInputFile(pdf_path, filename=f"{output_filename}.pdf")

    # Only send .md if rclone upload was not successful
    if not rclone_uploaded:
        md_file = FSInputFile(md_path, filename=f"{output_filename}.md")
        await message.answer_document(md_file, caption="Markdown transcript")

    await message.answer_document(pdf_file, caption="PDF transcript")

    # Show completion message (unified with other sources)
    if rclone_uploaded:
        await status_message.edit_text("Done! Your transcript is ready. (Markdown synced to Dropbox)")
    else:
        await status_message.edit_text("Done! Your transcript is ready.")


async def _process_audio_file(
    message: Message,
    file,
    file_name: str,
    status_message: Message,
    caption: str | None = None,
) -> None:
    """Process an uploaded audio file and send the transcript.

    Args:
        message: The Telegram message containing the audio file
        file: The audio file object
        file_name: Name of the audio file
        status_message: Status message to update with progress
        caption: Optional caption from the Telegram message for metadata context
    """
    # Get user settings
    chat_id = message.chat.id
    enable_translation = settings_store.get(chat_id, "translation", False)

    # Get user model preferences
    transcriber_model_key = settings_store.get(chat_id, "transcriber_model", DEFAULT_TRANSCRIBER_MODEL)
    editor_model_key = settings_store.get(chat_id, "editor_model", DEFAULT_EDITOR_MODEL)
    transcriber_model = MODELS[transcriber_model_key]
    editor_model = MODELS[editor_model_key]

    # Create config overrides if user selected different models
    transcriber_config = config.transcriber
    if transcriber_model != config.transcriber.model:
        transcriber_config = replace(config.transcriber, model=transcriber_model)

    editor_config = config.editor
    if editor_model != config.editor.model:
        editor_config = replace(config.editor, model=editor_model)

    # Sanitize filename to prevent path traversal attacks
    safe_filename = sanitize_filename(file_name)
    base_name = os.path.splitext(safe_filename)[0] or "audio"

    # Create a unique temporary directory for this request to prevent collisions
    request_id = uuid.uuid4().hex[:12]
    request_temp_dir = os.path.join(config.temp_dir, f"audio_{request_id}")
    os.makedirs(request_temp_dir, exist_ok=True)

    try:
        # Download file from Telegram
        await status_message.edit_text("Downloading audio file...")

        # Get file extension (sanitized)
        ext = os.path.splitext(safe_filename)[1] or ".mp3"
        audio_path = os.path.join(request_temp_dir, f"input{ext}")

        # Download file with permission fix for Local Bot API Server
        try:
            await message.bot.download(file, destination=audio_path)
        except PermissionError as e:
            # When using Local Bot API Server, files may have restricted permissions
            # Try to fix permissions and copy manually
            logger.warning(f"Permission error during download, attempting fix: {e}")

            # Get file info to find the local path
            file_info = await message.bot.get_file(file.file_id)
            if file_info.file_path:
                # Convert container path to host path if using local mode
                local_files_path = config.telegram.local_files_path
                if local_files_path and file_info.file_path.startswith("/var/lib/telegram-bot-api"):
                    host_path = file_info.file_path.replace(
                        "/var/lib/telegram-bot-api",
                        local_files_path,
                        1
                    )
                    logger.info(f"Attempting to fix permissions for: {host_path}")

                    # Try to make file readable (requires sudo or appropriate permissions)
                    try:
                        os.chmod(host_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                    except PermissionError:
                        # If direct chmod fails, try with sudo
                        subprocess.run(
                            ["sudo", "-n", "chmod", "644", host_path],
                            check=True,
                            capture_output=True
                        )

                    # Copy the file instead of using aiogram download
                    shutil.copy2(host_path, audio_path)
                    logger.info(f"Successfully copied file after permission fix")
                else:
                    raise  # Re-raise if not local mode
            else:
                raise  # Re-raise if no file_path

        # Create metadata from caption if provided
        transcription_metadata = None
        editor_metadata = None
        if caption:
            transcription_metadata = TranscriptionMetadata(
                source_name="Direct Upload",
                title=base_name,
                description=caption,
            )
            editor_metadata = VideoEditorMetadata(
                title=base_name,
                channel="Direct Upload",
                description=caption,
            )

        # Transcribe
        await status_message.edit_text("Transcribing audio...")
        raw_transcript = await transcribe(
            audio_path,
            transcriber_config,
            metadata=transcription_metadata,
            on_status=lambda s: logger.info(s),
        )

        # Edit/format
        await status_message.edit_text("Formatting transcript...")
        edited_transcript = await edit(
            raw_transcript,
            editor_config,
            enable_translation=enable_translation,
            metadata=editor_metadata,
            on_status=lambda s: logger.info(s),
        )

        # Generate output files
        await status_message.edit_text("Generating output files...")

        # Extract title from transcript for filename
        title = extract_title_from_transcript(edited_transcript)
        if title:
            # Sanitize the title for use as filename
            safe_title = sanitize_filename(title, max_length=200)
        else:
            # Fallback to original filename if no title found
            safe_title = base_name

        # Add date stamp
        date_stamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{safe_title} - {date_stamp}"

        # Save Markdown
        md_path = os.path.join(request_temp_dir, "transcript.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(edited_transcript)

        # Generate PDF
        pdf_path = os.path.join(request_temp_dir, "transcript.pdf")
        await generate_pdf(edited_transcript, pdf_path)

        # Upload to rclone if enabled for this user
        rclone_uploaded = await upload_to_rclone(md_path, f"{output_filename}.md", chat_id)

        # Send files
        await status_message.edit_text("Sending files...")

        pdf_file = FSInputFile(pdf_path, filename=f"{output_filename}.pdf")

        # Only send .md if rclone upload was not successful
        if not rclone_uploaded:
            md_file = FSInputFile(md_path, filename=f"{output_filename}.md")
            await message.answer_document(md_file, caption="Markdown transcript")

        await message.answer_document(pdf_file, caption="PDF transcript")

        if rclone_uploaded:
            await status_message.edit_text("Done! Your transcript is ready. (Markdown synced to Dropbox)")
        else:
            await status_message.edit_text("Done! Your transcript is ready.")

    finally:
        # Cleanup entire temp directory for this request
        try:
            shutil.rmtree(request_temp_dir)
        except Exception:
            pass
