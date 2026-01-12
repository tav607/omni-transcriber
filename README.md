# Omni Transcriber

Telegram bot for AI-powered audio transcription using Google Gemini API.

## Features

- Transcribe audio from YouTube, Bilibili, Apple Podcasts, and Xiaoyuzhou (小宇宙)
- Transcribe uploaded audio files (mp3, m4a, wav, webm, ogg, flac)
- Generate formatted transcripts with summary and key points
- Output as both Markdown and PDF files
- Chinese summary with original language transcript preservation
- User settings via Telegram commands:
  - `/model` - Choose AI model (Flash/Pro) for transcription and editing
  - `/translation` - Toggle inline Chinese translation for non-Chinese content

## Prerequisites

- Python 3.10+
- ffmpeg (for audio extraction)
- System libraries for WeasyPrint (PDF generation)

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repo-url>
cd omni_transcriber

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
| `TRANSCRIBER_MODEL` | `gemini-2.5-flash` | Model for transcription |
| `EDITOR_MODEL` | `gemini-2.5-pro` | Model for formatting |
| `TRANSCRIBER_TEMPERATURE` | `1.0` | Transcription temperature |
| `EDITOR_TEMPERATURE` | `1.0` | Editor temperature |
| `TRANSCRIBER_THINKING_LEVEL` | `low` | Thinking level: `low` or `high` |
| `EDITOR_THINKING_LEVEL` | `high` | Thinking level: `low` or `high` |
| `TEMP_DIR` | `/tmp/omni_transcriber` | Temporary file directory |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RSSHUB_BASE_URL` | *(empty)* | RSSHub instance URL (required for Xiaoyuzhou) |
| `RSSHUB_KEY` | *(empty)* | RSSHub access key (optional) |

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

The bot will reply with:
- A Markdown file containing the formatted transcript
- A PDF file for easy reading and sharing

### Commands

- `/start` - Welcome message
- `/help` - Usage instructions
- `/model` - Choose AI model (Flash/Pro) for transcriber and editor
- `/translation` - Toggle inline Chinese translation

## Proxy Support

The bot automatically detects and uses proxy from environment variables:
- `HTTPS_PROXY` / `https_proxy`
- `HTTP_PROXY` / `http_proxy`

## License

MIT
