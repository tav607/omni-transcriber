# Omni Transcriber

Telegram bot for AI-powered audio transcription using Google Gemini API.

## Features

- Transcribe audio from YouTube, Bilibili, Apple Podcasts, and Xiaoyuzhou (小宇宙)
- Transcribe uploaded audio files (mp3, m4a, mp4, wav, webm, ogg, flac)
- Support large audio files up to 2GB via Local Bot API Server
- Generate formatted transcripts with summary and key points
- Auto-detect podcast/interview videos and format them with takeaways, Q&A, and
  highlights (override per message with `#podcast` / `#nopodcast`)
- Output as both Markdown and PDF files
- Watch a Dropbox folder: drop a recording in, get the note in another folder, with
  per-folder settings in a `CONTEXT.md` (see Dropbox Folder Mode)
- Chinese summary with original language transcript preservation (Chinese text is
  normalized to Simplified script; Japanese is left alone)
- User settings via Telegram commands:
  - `/model` - Choose AI models for transcription and editing (Argon by default,
    or Flash)
  - `/translation` - Toggle inline Chinese translation for non-Chinese content
    (applies to podcasts and videos alike)

## Prerequisites

- Python 3.11+
- ffmpeg (for audio extraction)
- System libraries for WeasyPrint (PDF generation)

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repo-url>
cd omni-transcriber

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Install System Dependencies

**Ubuntu/Debian:**

```bash
# WeasyPrint dependencies
sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# ffmpeg for audio extraction
sudo apt install ffmpeg
```

**macOS:**

```bash
brew install pango libffi ffmpeg
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

**Required:**

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Get from [@BotFather](https://t.me/BotFather) |
| Vertex AI credentials | Not environment variables by default: the Vertex AI API key goes in `~/.vertex-key` and the GCP project id in `~/.vertex-project` (one bare value per file). `VERTEX_API_KEY` and `VERTEX_PROJECT` override the files when set. |

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_ALLOWED_CHAT_IDS` | *(empty, allows all)* | Comma-separated authorized Chat IDs |
| `TRANSCRIBER_MODEL` | `gemini-4-argon` | Model for transcription (Dropbox watcher; Telegram chats choose through `/model`) |
| `EDITOR_MODEL` | `gemini-4-argon` | Model for editing and the summary/metadata step (Dropbox watcher; Telegram chats choose through `/model`) |
| `TRANSLATION_MODEL` | `gemini-3.8-flash` | Model for inline translation and the cross-chunk glossary |
| `TRANSCRIBER_TEMPERATURE` | `1.0` | Transcription temperature |
| `EDITOR_TEMPERATURE` | `1.0` | Editor temperature |
| `TRANSCRIBER_THINKING_LEVEL` | `high` | Thinking level: `low`, `medium`, or `high` |
| `EDITOR_THINKING_LEVEL` | `high` | Thinking level: `low`, `medium`, or `high` |
| `GOOGLE_VERTEX_BASE_URL` | *(official endpoint)* | Route Gemini (Vertex AI) through a relay/proxy (read by the google-genai SDK) |
| `TEMP_DIR` | `/tmp/omni_transcriber` | Temporary file directory |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RSSHUB_BASE_URL` | *(empty)* | RSSHub instance URL (required for Xiaoyuzhou) |
| `RSSHUB_KEY` | *(empty)* | RSSHub access key (optional) |
| `TELEGRAM_API_SERVER` | *(empty)* | Local Bot API server URL (for files > 20MB) |
| `TELEGRAM_LOCAL_FILES_PATH` | *(empty)* | Host path for Local Bot API files |

### 4. Run the Bot

```bash
source .venv/bin/activate
python -m src.main
```

### 5. Get Your Chat ID

To restrict bot access, you need your Telegram Chat ID:

1. Start the bot without `TELEGRAM_ALLOWED_CHAT_IDS` set
2. Send any message to the bot
3. Check the logs for `Unauthorized access attempt from chat_id: XXXXXX`
4. Add that ID to `TELEGRAM_ALLOWED_CHAT_IDS` in `.env`

## Usage

- **YouTube**: Send a YouTube URL (youtube.com, youtu.be, shorts)
- **Bilibili**: Send a Bilibili URL (bilibili.com, b23.tv)
- **Apple Podcasts**: Send an Apple Podcasts URL (podcasts.apple.com)
- **Xiaoyuzhou**: Send a Xiaoyuzhou URL (xiaoyuzhoufm.com) - requires RSSHub
- **Audio file**: Send an audio file directly

Apple Podcasts and Xiaoyuzhou always use podcast mode. YouTube/Bilibili videos
are auto-classified: interview / talk-show / panel content is formatted as a
podcast (Info, Summary, Takeaways, Q&A, Highlights), everything else gets the
generic title/summary/key-points format. Add `#podcast` or `#nopodcast` in the
same message as the link to force the choice.

The bot will reply with:
- A Markdown file containing the formatted transcript
- A PDF file for easy reading and sharing

### Commands

- `/start` - Welcome message
- `/help` - Usage instructions
- `/model` - Choose AI models for the transcriber and the editor (Argon or Flash)
- `/translation` - Toggle inline Chinese translation

## Dropbox Folder Mode

With the `DROPBOX_*` variables set, the bot also watches a Dropbox folder. Drop an audio file anywhere under it and the transcript note is uploaded to `DROPBOX_OUTPUT_FOLDER`, the PDF is sent to `DROPBOX_TELEGRAM_CHAT_ID`, and the audio moves to a `_processed` subfolder next to where it was.

| Variable | Default | Description |
|----------|---------|-------------|
| `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_REFRESH_TOKEN` | | Dropbox app credentials, used to detect changes (metadata access is enough) |
| `DROPBOX_WATCH_FOLDER` | unset (off) | Folder to watch; empty string for the app folder root |
| `DROPBOX_RCLONE_REMOTE`, `DROPBOX_RCLONE_BASE_PATH` | `dropbox`, empty | rclone remote that moves the files, and the path the watch folder has on it |
| `DROPBOX_OUTPUT_FOLDER` | | Folder on the rclone remote for the notes |
| `DROPBOX_TELEGRAM_CHAT_ID` | | Chat that gets status messages and the PDF |
| `DROPBOX_PROCESSED_SUBFOLDER`, `DROPBOX_FAILED_SUBFOLDER` | `_processed`, `_failed` | Where finished and failed audio moves |
| `DROPBOX_POLL_INTERVAL` | `30` | Longpoll timeout in seconds |
| `DROPBOX_MAX_FILE_SIZE_MB` | `500` | Larger files fail without being downloaded |

- Every audio file outside `_processed` and `_failed` is work to do, including files that arrived while the service was down: the folder is scanned at startup.
- A file that fails moves to `_failed`, with the reason sent to Telegram. Move it back out to retry.
- The note is named `<title> - <recording date>`. The recording date is the date in the file name if there is one, otherwise the file's own modification date. A name already taken gets ` (2)`, ` (3)`, ... instead of being overwritten.
- The note starts with YAML front matter: `source` (the audio file), `recorded`, `mode`.
- The first status message lists the settings the run uses.

### CONTEXT.md

A folder's settings live in a `CONTEXT.md`: YAML front matter, then free-text background.

```markdown
---
mode: meeting
participants:
  - Jane: the client, an analyst in New York
  - Sam: my manager, mostly listening
glossary: [CoWoS, CCL]
---
Weekly calls with the client about semiconductor research.
```

Every `CONTEXT.md` from the watch folder root down to the audio file's folder applies, root first. Put what all folders share in the root file. A folder overrides a setting by stating it; `glossary` terms and background text add up instead. HTML comments in the background are dropped.

| Setting | Values | Effect |
|---------|--------|--------|
| `mode` | `meeting` (default), `earnings`, `generic` | Picks the speaker-labeling rule, the edit prompt and the default sections. `meeting`: every turn labeled by name or `Speaker N`, English terms kept in Latin spelling. `earnings`: `Operator` plus real names, and the transcript split into Management Discussion and Q&A. `generic`: single-track audio, speaker changes only where obvious. |
| `participants` | list of `Name` or `Name: description` | People expected on the recording. Their turns get exactly these labels once they are recognized; others are labeled as the mode says. |
| `roles` | list of two or more labels, asking side first | Fixed labels for an interview, e.g. `[BDA, 专家]`. Every turn gets one of them and real names are never used. Cannot be combined with `participants`. |
| `glossary` | list of terms | Spellings the editor must keep. |
| `translation` | `true` / `false` | Inline Chinese translation for non-Chinese transcripts. |
| `output_folder` | subfolder path | Subfolder of `DROPBOX_OUTPUT_FOLDER` for this folder's notes. |
| `sections` | list, see below | The note's sections, replacing the mode's defaults. |

Each entry of `sections` has a `key`, a `prompt` saying what to write, an optional `type` (`list`, the default; `text`; `qa` for question and answer pairs) and an optional `heading`:

```yaml
sections:
  - key: summary
    type: text
    prompt: 访谈摘要 (中文，不超过300字)
  - key: data_points
    heading: Data Points
    prompt: 专家提到的具体数字，保留原始数值与单位
```

A title is always written; add a `title` entry only to change how it is worded. A section the model leaves empty is left out of the note.

A `CONTEXT.md` that cannot be read or contains a mistake (unknown setting, bad value, the pre-2026-11 `## Background` / `## Sections` format) stops the files it governs: the reason goes to Telegram, the audio stays where it is, and it runs once a `CONTEXT.md` is saved again.

## Local Bot API Server (Optional)

By default, Telegram Bot API limits file downloads to 20MB. To handle larger audio files (up to 2GB), deploy a Local Bot API Server:

### 1. Get Telegram API Credentials

1. Go to https://my.telegram.org/auth
2. Log in with your phone number
3. Select "API development tools"
4. Create an app to get `api_id` and `api_hash`

### 2. Configure and Run

```bash
# Add to .env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_API_SERVER=http://localhost:8081
TELEGRAM_LOCAL_FILES_PATH=/absolute/path/to/data/telegram-bot-api

# Start the server
docker compose up -d
```

### 3. File Permission Setup

The Docker container runs as `messagebus` user and creates files with restricted permissions. To allow the bot to read downloaded files:

```bash
# Add your user to messagebus group
sudo usermod -a -G messagebus $USER

# Log out and back in, or use newgrp
newgrp messagebus
```

The bot is managed by systemd with `SupplementaryGroups=messagebus` for proper group permissions:

```bash
sudo systemctl start omni-transcriber
sudo systemctl status omni-transcriber
journalctl -u omni-transcriber -f
```

### 4. First-time Setup

If the bot was previously running with official Telegram API, log out first:

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# Wait ~10 minutes before starting the bot
```

## Proxy Support

The bot automatically detects and uses proxy from environment variables:
- `HTTPS_PROXY` / `https_proxy`
- `HTTP_PROXY` / `http_proxy`

## License

MIT
