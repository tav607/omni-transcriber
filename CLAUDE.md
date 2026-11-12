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
   **or** Audio file upload → downloaded from Telegram. An uploaded or watched file always carries a date into the transcription prompt for the year anchor (`recording_date`): a date in the filename, otherwise the day the file arrived, worded as "on or before"
2. **Transcription**: `services/transcriber.py` sends each chunk's audio inline in the request (Vertex AI has no File API; a chunk over 50MB, which only uncompressed audio reaches, is re-encoded to 64kbps mono first) and transcribes with the Gemini model (a failing call climbs the retry ladder, then moves to `gemini-3.1-pro-preview`; a response blocked with `block_reason` set, OTHER so far, moves there on the first block without retrying; a recitation block (`finish_reason=RECITATION`, output matching published text) skips both the retries and the model switch, and the whole audio is cut again into 10-minute chunks and transcribed once more; every retry and failure line names the chunk and model, e.g. `Transcription chunk 2/4 on gemini-4-argon failed (attempt 1/6)`); each chunk is folded to simplified Chinese (zhconv, text with Japanese kana left alone) before the chunks are merged. Audio is cut into 15-minute chunks through the middle of the longest pause in the minute before each mark (20 ms frame levels, threshold relative to the stretch itself), so chunks share no audio and those seams are joined as is; only a minute without a 0.3-second pause falls back to cutting at the mark with 10 seconds of shared audio. Such a seam is found by an endpoint-anchored alignment of the previous chunk's tail against the next chunk's head (both backported from the transcribe skill), which never consumes a new speaker turn and keeps a short duplicate rather than delete past the real seam (skips over 400 characters are ignored)
3. **Editing**: `services/editor.py` formats raw transcript into structured Markdown (Chinese summary + original language transcript); a chunk with nothing spoken in it (only timestamps, sound tags, labels) is dropped and one with under 50 characters of speech is kept raw without an edit call, in both the generic and the podcast editor; chunks are cut at speaker turns and sized evenly, at most 32,000 characters or 20,000 for CJK-heavy transcripts (Argon's 32,768-token output cap also has to hold its thinking); every edited chunk first goes through the edit fidelity guard (`services/edit_guard.py`), which aligns it against its raw input and puts back whatever the editor dropped or reworded beyond cleanup (fillers, stutters, restated false starts, small function-word trims and formatting variants survive; numbers and negations never count as cleanup); after the guard, each chunk's number tokens are compared with its raw input and a figure that vanished or appeared is logged as `Editor changed numbers in chunk N` (the edit prompts forbid changing numbers; this reports what still got through); once the edited chunks are joined, `_normalize_speaker_labels` merges name variants and renumbers `**Speaker N:**` labels by first appearance, in both editors; the final Markdown and title are folded to simplified Chinese again, since metadata and translation calls can answer in traditional script
4. **Output**: `services/pdf_generator.py` converts Markdown to PDF via WeasyPrint

### Dropbox Folder Mode

`src/watcher/` processes audio dropped into the watched Dropbox folder. Settings come from `CONTEXT.md` files (format and settings table in the README).

- `context_loader.py` parses a `CONTEXT.md` (YAML front matter plus background text) and folds every one from the watch root down to the audio's folder into a `ContextConfig`. Anything unreadable or invalid raises `ContextError`; there are no silent defaults. The earlier format failed quietly in three ways: a failed download counted as "no file", a misspelled heading was ignored, and the HTML comments of an unfilled template went to the model as background.
- `services/sections.py` holds output sections as data (`Section`: key, prompt, type, heading). One list produces the metadata prompt, the response schema and the rendered Markdown, so the Telegram default, the mode defaults and a folder's custom `sections` all take the same path, always with a schema.
- `mode`, `roles` and `participants` reach transcription as a `Labeling` (`_speaker_guideline` in `transcriber.py`) and editing as `mode` / `roles` arguments of `edit()` (`_edit_system_prompt`). The meeting and earnings guidelines, the roster rule, the role-swap fix and the two edit prompts are the transcribe skill's, word for word; the `participants` rule is this repo's own and has not been measured.
- `dropbox_watcher.py`: the folder is listed at startup and every audio file outside `_processed` / `_failed` is scheduled, at most `MAX_CONCURRENT_FILES` at once. A `ContextError` holds the file back (`blocked`), reports it, and a later `CONTEXT.md` change retries it. A Dropbox `ApiError` in the loop takes a fresh listing, since a rejected cursor never recovers.
- The Dropbox app token reads metadata only (`files/download` is refused), so whether a `CONTEXT.md` exists comes from `files_get_metadata` and its content from `rclone cat`.
- `processor.py`: the file name and recording date go to the summary step as well as to transcription (`RecordingInfo` in `editor.py`); without them a 2026 earnings clip was titled "Q2 2025" and a file the owner had named was retitled from its content. A failed file moves to `_failed`; the note is named by recording date, never overwrites an existing note, and carries `source` / `recorded` / `mode` front matter (the PDF does not).

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
- `TELEGRAM_BOT_TOKEN` - Required
- Gemini runs on Vertex AI (moved from AI Studio on 2026-11-11, when AI Studio went prepaid). Its credentials are not in `.env`: the API key is read from `~/.vertex-key` and the GCP project id from `~/.vertex-project` (`VERTEX_API_KEY` / `VERTEX_PROJECT` override the files; the offline tests set them). `utils/gemini.py` builds the one client (`get_client`) and names every model by full path on the global endpoint (`model_path`), because a key alone is routed to us-central1, which does not serve `gemini-4-argon`. Vertex has no `-latest` aliases, so every model id is concrete (the fallback model is `gemini-3.1-pro-preview`). `/model` offers Flash and Argon; the Pro choice was removed on 2026-11-11, and a chat whose stored choice is no longer offered reads as the default
- `TRANSCRIBER_MODEL` / `EDITOR_MODEL` / `TRANSLATION_MODEL` - Gemini models (defaults: gemini-4-argon for transcription and editing, gemini-3.8-flash for translation). `EDITOR_MODEL` also runs the summary/metadata step; the cross-chunk glossary and inline translation use `TRANSLATION_MODEL`. Argon transcribes since 2026-10-28: paired against 3.8-flash in the transcribe skill it scored WER 3.97% against 4.54% on a 65-minute earnings call, 6.86% against 7.35% on a 37-minute interview and CER 19.8% against 22.8% on a 38-minute Mandarin meeting, was never blocked in 79 calls, and finished each 15-minute chunk 2-3 times faster; long audio costs about the same per hour at its introductory price and about twice 3.8's once that doubles on 2026-12-01. Argon edits because, re-editing three October 2026 expert calls, it lost 2-5% of distinct content words before the fidelity guard where 3.8-flash lost 4-15%, at about the same cost per call. All three are pinned to a version on purpose: `gemini-flash-latest` repointed to 3.7-flash on its release day, so an alias here would swap the model without anyone deciding to. A pin can still go stale: since 2026-10-08 Google answers `gemini-3.7-flash` requests with 3.8-flash, visible only in the response's `model_version`. Every Gemini call therefore compares `model_version` with the requested id and logs one WARNING per rerouted pair (`Requested X but Google served Y`; `*-latest` aliases and version suffixes such as `-001` are not reroutes). There is no admin channel, so this only reaches the journal: `journalctl -u omni-transcriber | grep "Google served"`. Telegram requests ignore `TRANSCRIBER_MODEL` / `EDITOR_MODEL` and resolve the chat's choice through `MODELS` in `src/bot/handlers.py` (flash/pro/argon for both, default argon); only the Dropbox watcher reads those two variables. A model change needs both places.
- `TRANSCRIBER_THINKING_LEVEL` / `EDITOR_THINKING_LEVEL` - Gemini native thinking_level: "low", "medium", or "high" (both default "high"; metadata generation is always "high", translation/glossary always "low")
- `TELEGRAM_ALLOWED_CHAT_IDS` - Comma-separated list for access control
- `RSSHUB_BASE_URL` / `RSSHUB_KEY` - Required for Xiaoyuzhou podcast support
- `YOUTUBE_SOURCE_ADDRESS` - Local address yt-dlp binds to for YouTube requests only (`_youtube_egress_opts` in `downloader.py`). YouTube answers every request from gcp-vm's Google address with "Sign in to confirm you're not a bot" (first seen 2026-11-12; a JS runtime and other player clients did not change it), so on gcp-vm this is the WARP tunnel's address `172.16.0.2`, and `/etc/wireguard/warp.conf` carries a matching `ip rule ... from 172.16.0.2 lookup 51888`. Without that rule the bind sends packets out the default interface and every YouTube request times out. The same error with the variable set means the WARP exit is blocked too. yt-dlp needs a JavaScript runtime for YouTube, which the `deno` dependency puts in `.venv/bin`; the service must be started through `uv run` (or with `.venv/bin` on `PATH`) for yt-dlp to find it.
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
