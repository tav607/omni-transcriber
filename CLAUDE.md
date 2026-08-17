# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the bot
uv run python -m src.main

# System dependencies (Ubuntu/Debian)
sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info ffmpeg
```

## Architecture

Telegram bot that transcribes audio (from YouTube/Bilibili/Apple Podcasts/Xiaoyuzhou URLs or uploaded files) using Google Gemini API, then outputs formatted Markdown and PDF files.

### Processing Pipeline

1. **Input**: YouTube/Bilibili URL → `services/downloader.py` downloads audio via yt-dlp
   **or** Apple Podcasts URL → `services/rss_parser.py` resolves RSS feed via iTunes Lookup API and locates the episode (by iTunes ID, then URL-slug title match) → `services/downloader.py` downloads from the RSS audio URL
   **or** Xiaoyuzhou URL → `services/xiaoyuzhou_parser.py` fetches audio URL via RSSHub → `services/downloader.py` downloads
   **or** Audio file upload → downloaded from Telegram
2. **Transcription**: `services/transcriber.py` uploads audio to Gemini File API, transcribes with Gemini model
3. **Editing**: `services/editor.py` formats raw transcript into structured Markdown (Chinese summary + original language transcript)
4. **Output**: `services/pdf_generator.py` converts Markdown to PDF via WeasyPrint

### Key Modules

- `src/config.py` - Configuration from environment variables, includes editor system prompt
- `src/bot/handlers.py` - Telegram message handlers, orchestrates the pipeline, manages user settings
- `src/bot/bot.py` - Bot initialization, command registration (whitelist-aware)
- `src/bot/middleware.py` - Chat ID authorization middleware
- `src/services/downloader.py` - Audio download via yt-dlp (YouTube/Bilibili/direct URLs)
- `src/services/rss_parser.py` - Apple Podcasts RSS feed lookup and episode matching (iTunes ID + slug-based title match)
- `src/services/xiaoyuzhou_parser.py` - Xiaoyuzhou podcast metadata extraction via RSSHub
- `src/utils/retry.py` - Retry wrapper for API calls
- `src/utils/url_parser.py` - YouTube/Bilibili/Apple Podcasts/Xiaoyuzhou URL detection

### Configuration

All config via environment variables (see `.env.example`). Key settings:
- `GEMINI_API_KEY` / `TELEGRAM_BOT_TOKEN` - Required
- `TRANSCRIBER_MODEL` / `EDITOR_MODEL` / `TRANSLATION_MODEL` - Gemini models (default: gemini-3.7-flash for all three). All three are pinned to a version on purpose: `gemini-flash-latest` repointed to 3.7-flash on its release day, so an alias here would swap the model without anyone deciding to.
- `TRANSCRIBER_THINKING_LEVEL` / `EDITOR_THINKING_LEVEL` - Gemini native thinking_level: "low", "medium", or "high" (both default "high"; metadata generation is always "high", translation/glossary always "low")
- `TELEGRAM_ALLOWED_CHAT_IDS` - Comma-separated list for access control
- `RSSHUB_BASE_URL` / `RSSHUB_KEY` - Required for Xiaoyuzhou podcast support
- `TELEGRAM_API_SERVER` - Local Bot API server URL (optional, for files > 20MB)

### Local Bot API Server (Optional)

For handling audio files larger than 20MB (Telegram's default limit), deploy a Local Bot API Server using `docker-compose.yml`. Requires:
- `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` from https://my.telegram.org/auth
- `TELEGRAM_API_SERVER` - Local Bot API server URL (e.g., `http://localhost:8081`)
- `TELEGRAM_LOCAL_FILES_PATH` - Host path matching the Docker volume mount (e.g., `/path/to/data/telegram-bot-api`)

**File Permission Handling**: The Docker container runs as `messagebus` user and creates files with `640` permissions. To allow the bot to read these files:
1. Add bot user to `messagebus` group: `sudo usermod -a -G messagebus $USER`
2. The systemd service uses `SupplementaryGroups=messagebus` for group permissions
3. Restart service: `sudo systemctl restart omni-transcriber`
