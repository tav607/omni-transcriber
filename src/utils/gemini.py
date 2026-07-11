"""Shared helpers for inspecting Gemini API responses."""

import asyncio

# Cap total in-flight Gemini generate_content/upload calls across every fan-out
# (transcription upload + chunk waves, editor edit/podcast-edit/translation). The
# bot and the Dropbox watcher share one event loop, so a per-run cap would still
# stack; a single module-level gate bounds the whole process. It also keeps the
# per-call wait_for budgets honest: without a gate, chunks queue in the default
# thread pool with their timeout clock already running.
MAX_CONCURRENT_GEMINI_CALLS = 6
# Safe at module scope on Python 3.10+: an asyncio.Semaphore binds to the loop on
# first await, and the app runs a single asyncio.run loop.
_gemini_sem = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_CALLS)


def is_truncated(response) -> bool:
    """True if the model stopped because it hit the max output token cap."""
    try:
        for cand in (response.candidates or []):
            fr = getattr(cand, "finish_reason", None)
            if fr is None:
                continue
            name = getattr(fr, "name", None) or str(fr)
            if "MAX_TOKENS" in name.upper():
                return True
    except Exception:
        pass
    return False
