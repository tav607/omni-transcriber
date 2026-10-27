"""Offline unit tests: no test calls a Gemini or Telegram API.

src.config loads its settings when first imported and refuses to start
without the two credentials, so placeholder values are set here, before any
test module imports src. Run from the repo root:

    uv run python -m unittest discover -s tests -t .
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:test-token")
