"""Offline unit tests: no test calls a Gemini or Telegram API.

src.config loads its settings when first imported and refuses to start
without the two credentials, so placeholder values are set here, before any
test module imports src. Run from the repo root:

    uv run python -m unittest discover -s tests -t .
"""

import os
import warnings

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:test-token")

# zhconv leaves its dictionary file open when it loads it on first use; load it
# here once so unittest's ResourceWarning does not clutter the test output.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ResourceWarning)
    try:
        from zhconv import convert

        convert("測", "zh-cn")
    except ImportError:
        pass
