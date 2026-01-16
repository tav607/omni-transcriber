import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer, FilesPathWrapper
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, BotCommandScopeChat, BotCommandScopeDefault

from ..config import config
from .middleware import AuthorizationMiddleware
from .handlers import router

logger = logging.getLogger(__name__)


# Default container path for Local Bot API Server
LOCAL_BOT_API_CONTAINER_PATH = "/var/lib/telegram-bot-api"


class DockerPathWrapper(FilesPathWrapper):
    """
    Map Docker container paths to host paths for Local Bot API Server.

    When using Local Bot API Server in Docker, file paths returned by getFile
    are container paths (e.g., /var/lib/telegram-bot-api/...). This wrapper
    converts them to host paths where files are actually accessible.
    """

    def __init__(self, container_path: str, host_path: str):
        self.container_path = container_path
        self.host_path = host_path

    def to_local(self, path: Path | str) -> Path | str:
        """Convert container path to host path."""
        path_str = str(path)
        if path_str.startswith(self.container_path):
            return path_str.replace(self.container_path, self.host_path, 1)
        return path

    def to_server(self, path: Path | str) -> Path | str:
        """Convert host path to container path."""
        path_str = str(path)
        if path_str.startswith(self.host_path):
            return path_str.replace(self.host_path, self.container_path, 1)
        return path


def get_proxy_url() -> str | None:
    """Get proxy URL from environment variables."""
    return (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )


def create_bot() -> Bot:
    """Create and configure the Telegram bot."""
    if not config.telegram.bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not configured")

    proxy_url = get_proxy_url()
    session = None

    api_server = config.telegram.api_server
    local_files_path = config.telegram.local_files_path

    if api_server or proxy_url:
        # Prepare TelegramAPIServer kwargs
        api_server_kwargs = {}

        # Enable local mode with path mapping if local_files_path is configured
        if api_server and local_files_path:
            api_server_kwargs["is_local"] = True
            api_server_kwargs["wrap_local_file"] = DockerPathWrapper(
                container_path=LOCAL_BOT_API_CONTAINER_PATH,
                host_path=local_files_path,
            )
            logger.info(
                f"Local Bot API mode enabled with path mapping: "
                f"{LOCAL_BOT_API_CONTAINER_PATH} -> {local_files_path}"
            )

        session = AiohttpSession(
            api=TelegramAPIServer.from_base(api_server, **api_server_kwargs) if api_server else None,
            proxy=proxy_url,
        )
        if api_server:
            logger.info(f"Using local Bot API server: {api_server}")
        if proxy_url:
            logger.info(f"Using proxy: {proxy_url}")

    return Bot(
        token=config.telegram.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        session=session,
    )


def create_dispatcher() -> Dispatcher:
    """Create and configure the dispatcher with middleware and handlers."""
    dp = Dispatcher()

    # Register authorization middleware
    if config.telegram.allowed_chat_ids:
        dp.message.middleware(
            AuthorizationMiddleware(config.telegram.allowed_chat_ids)
        )
        logger.info(
            f"Authorization middleware enabled for chat IDs: "
            f"{config.telegram.allowed_chat_ids}"
        )
    else:
        logger.warning(
            "No allowed chat IDs configured! Bot is open to everyone. "
            "Set TELEGRAM_ALLOWED_CHAT_IDS to restrict access."
        )

    # Register handlers
    dp.include_router(router)

    return dp


async def run_bot():
    """Run the bot with polling."""
    bot = create_bot()
    dp = create_dispatcher()

    # Register bot commands for the menu
    commands = [
        BotCommand(command="start", description="Start the bot"),
        BotCommand(command="help", description="Show help information"),
        BotCommand(command="model", description="Choose AI model (Flash/Pro)"),
        BotCommand(command="translation", description="Toggle translation mode (on/off)"),
    ]

    # Set up command visibility based on whitelist
    if config.telegram.allowed_chat_ids:
        # Clear commands for everyone (non-whitelisted users see nothing)
        await bot.set_my_commands([], scope=BotCommandScopeDefault())
        logger.info("Cleared default command menu for non-whitelisted users")

        # Set commands for each allowed chat
        for chat_id in config.telegram.allowed_chat_ids:
            try:
                await bot.set_my_commands(
                    commands,
                    scope=BotCommandScopeChat(chat_id=chat_id),
                )
                logger.info(f"Set commands for chat_id: {chat_id}")
            except Exception as e:
                # May fail if bot hasn't chatted with user yet
                logger.debug(f"Could not set commands for chat_id {chat_id}: {e}")
    else:
        # No whitelist - show commands to everyone
        await bot.set_my_commands(commands)
        logger.info("Bot commands registered for all users")

    logger.info("Starting bot...")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
