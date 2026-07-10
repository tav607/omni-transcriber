"""Shared helpers for inspecting Gemini API responses."""


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
