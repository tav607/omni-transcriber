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
- Chinese summary with original language transcript preservation
- User settings via Telegram commands:
  - `/model` - Choose AI model (Flash/Pro) for transcription and editing
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
| `GEMINI_API_KEY` | Get from [Google AI Studio](https://aistudio.google.com/apikey) |

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_ALLOWED_CHAT_IDS` | *(empty, allows all)* | Comma-separated authorized Chat IDs |
| `TRANSCRIBER_MODEL` | `gemini-pro-latest` | Model for transcription |
| `EDITOR_MODEL` | `gemini-pro-latest` | Model for formatting |
| `TRANSLATION_MODEL` | `gemini-flash-latest` | Model for inline translation |
| `TRANSCRIBER_TEMPERATURE` | `1.0` | Transcription temperature |
| `EDITOR_TEMPERATURE` | `1.0` | Editor temperature |
| `TRANSCRIBER_THINKING_LEVEL` | `low` | Thinking level: `low` or `high` |
| `EDITOR_THINKING_LEVEL` | `high` | Thinking level: `low` or `high` |
| `GOOGLE_GEMINI_BASE_URL` | *(official endpoint)* | Route Gemini through a relay/proxy (read by the google-genai SDK) |
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
- `/model` - Choose AI model (Flash/Pro) for transcriber and editor
- `/translation` - Toggle inline Chinese translation

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
