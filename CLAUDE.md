# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the bot
uv run python -m src.main

# Run the offline unit tests (no API calls; tests/__init__.py sets placeholder credentials)
uv run python -m unittest discover -s tests -t .

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
2. **Transcription**: `services/transcriber.py` uploads audio to Gemini File API, transcribes with Gemini model (a chunk whose upload failed, or whose file handle comes back 403, is sent with its audio inline instead, skipping the retry ladder and model fallback; a response blocked with `block_reason` set, OTHER so far, moves to `gemini-pro-latest` on the first block and is never retried inline, since the block follows the request; every retry and failure line names the chunk and model, e.g. `Transcription chunk 2/4 on gemini-4-argon failed (attempt 1/6)`); each chunk is folded to simplified Chinese (zhconv, text with Japanese kana left alone) before the chunks are merged. Audio is cut into 15-minute chunks through the middle of the longest pause in the minute before each mark (20 ms frame levels, threshold relative to the stretch itself), so chunks share no audio and those seams are joined as is; only a minute without a 0.3-second pause falls back to cutting at the mark with 10 seconds of shared audio. Such a seam is found by an endpoint-anchored alignment of the previous chunk's tail against the next chunk's head (both backported from the transcribe skill), which never consumes a new speaker turn and keeps a short duplicate rather than delete past the real seam (skips over 400 characters are ignored)
3. **Editing**: `services/editor.py` formats raw transcript into structured Markdown (Chinese summary + original language transcript); a chunk with nothing spoken in it (only timestamps, sound tags, labels) is dropped and one with under 50 characters of speech is kept raw without an edit call, in both the generic and the podcast editor; chunks are cut at speaker turns and sized evenly, at most 32,000 characters or 20,000 for CJK-heavy transcripts (Argon's 32,768-token output cap also has to hold its thinking); every edited chunk first goes through the edit fidelity guard (`services/edit_guard.py`), which aligns it against its raw input and puts back whatever the editor dropped or reworded beyond cleanup (fillers, stutters, restated false starts, small function-word trims and formatting variants survive; numbers and negations never count as cleanup); the final Markdown and title are folded to simplified Chinese again, since metadata and translation calls can answer in traditional script
4. **Output**: `services/pdf_generator.py` converts Markdown to PDF via WeasyPrint

### Key Modules

- `src/config.py` - Configuration from environment variables, includes editor system prompt
- `src/bot/handlers.py` - Telegram message handlers, orchestrates the pipeline, manages user settings
- `src/bot/bot.py` - Bot initialization, command registration (whitelist-aware)
- `src/bot/middleware.py` - Chat ID authorization middleware
- `src/services/downloader.py` - Audio download via yt-dlp (YouTube/Bilibili/direct URLs)
- `src/services/rss_parser.py` - Apple Podcasts RSS feed lookup and episode matching (iTunes ID + slug-based title match)
- `src/services/xiaoyuzhou_parser.py` - Xiaoyuzhou podcast metadata extraction via RSSHub
- `src/services/edit_guard.py` - Edit fidelity guard (`_restore_edit_losses`), backported from the transcribe skill; runs in a worker thread because a 32k-character Chinese chunk takes about 1.5s to align
- `src/utils/gemini.py` - Process-wide Gemini concurrency gate and response checks (truncation, served-model reroutes)
- `src/utils/chinese.py` - Simplified/traditional folding (`_to_simplified`, optional zhconv import that degrades to a no-op)
- `src/utils/retry.py` - Retry wrapper for API calls
- `src/utils/url_parser.py` - YouTube/Bilibili/Apple Podcasts/Xiaoyuzhou URL detection

### Configuration

All config via environment variables (see `.env.example`). Key settings:
- `GEMINI_API_KEY` / `TELEGRAM_BOT_TOKEN` - Required
- `TRANSCRIBER_MODEL` / `EDITOR_MODEL` / `TRANSLATION_MODEL` - Gemini models (defaults: gemini-4-argon for transcription and editing, gemini-3.8-flash for translation). `EDITOR_MODEL` also runs the summary/metadata step; the cross-chunk glossary and inline translation use `TRANSLATION_MODEL`. Argon transcribes since 2026-10-28: paired against 3.8-flash in the transcribe skill it scored WER 3.97% against 4.54% on a 65-minute earnings call, 6.86% against 7.35% on a 37-minute interview and CER 19.8% against 22.8% on a 38-minute Mandarin meeting, was never blocked in 79 calls, and finished each 15-minute chunk 2-3 times faster; long audio costs about the same per hour at its introductory price and about twice 3.8's once that doubles on 2026-12-01. Argon edits because, re-editing three October 2026 expert calls, it lost 2-5% of distinct content words before the fidelity guard where 3.8-flash lost 4-15%, at about the same cost per call. All three are pinned to a version on purpose: `gemini-flash-latest` repointed to 3.7-flash on its release day, so an alias here would swap the model without anyone deciding to. A pin can still go stale: since 2026-10-08 Google answers `gemini-3.7-flash` requests with 3.8-flash, visible only in the response's `model_version`. Every Gemini call therefore compares `model_version` with the requested id and logs one WARNING per rerouted pair (`Requested X but Google served Y`; `*-latest` aliases and version suffixes such as `-001` are not reroutes). There is no admin channel, so this only reaches the journal: `journalctl -u omni-transcriber | grep "Google served"`. Telegram requests ignore `TRANSCRIBER_MODEL` / `EDITOR_MODEL` and resolve the chat's choice through `MODELS` in `src/bot/handlers.py` (flash/pro/argon for both, default argon); only the Dropbox watcher reads those two variables. A model change needs both places.
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
