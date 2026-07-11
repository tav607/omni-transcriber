import asyncio
import logging
import os
import shutil

from .config import config, setup_logging
from .bot.bot import run_bot
from .watcher import run_dropbox_watcher


def check_external_dependencies() -> None:
    """Check that required external tools are installed.

    Raises:
        RuntimeError: If required dependencies are missing.
    """
    missing = []

    if not shutil.which("ffprobe"):
        missing.append("ffprobe")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")

    if missing:
        raise RuntimeError(
            f"Required external dependencies not found: {', '.join(missing)}. "
            "Install ffmpeg (which includes ffprobe), e.g. `apt install ffmpeg`."
        )


async def main_async():
    """Run the bot and optional watcher concurrently.

    Uses TaskGroup to ensure that if any task fails with an exception,
    other tasks are cancelled and the exception is propagated immediately.
    This is important because run_bot() runs indefinitely - without TaskGroup,
    asyncio.gather with return_exceptions=True would never return even if
    the watcher crashes.
    """
    logger = logging.getLogger(__name__)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_bot(), name="bot")

        if config.dropbox_watcher.is_enabled:
            logger.info("Dropbox watcher is enabled, starting in parallel")
            tg.create_task(run_dropbox_watcher(), name="dropbox_watcher")
        else:
            logger.info("Dropbox watcher is disabled")


def main():
    """Main entry point for the application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Fail fast if ffmpeg/ffprobe are missing: as a long-running service this
    # surfaces the deployment problem at startup instead of letting every
    # transcription task hit a raw subprocess traceback mid-run.
    check_external_dependencies()

    # Ensure temp directory exists
    os.makedirs(config.temp_dir, exist_ok=True)

    # Log configuration
    logger.info("Starting Omni Transcriber Bot")
    logger.info(f"Temp directory: {config.temp_dir}")
    logger.info(f"Transcriber model: {config.transcriber.model}")
    logger.info(f"Editor model: {config.editor.model}")
    logger.info(
        f"Allowed chat IDs: {config.telegram.allowed_chat_ids or 'ALL (no restriction)'}"
    )

    if config.dropbox_watcher.is_enabled:
        logger.info(f"Dropbox watch folder: {config.dropbox_watcher.watch_folder}")
        logger.info(f"Dropbox output folder: {config.dropbox_watcher.output_folder}")

    # Run the bot (and optionally watcher)
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
