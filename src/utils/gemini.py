"""Shared helpers for inspecting Gemini API responses."""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Cap total in-flight Gemini generate_content/upload calls across every fan-out
# (transcription upload + chunk waves, editor edit/podcast-edit/translation). The
# bot and the Dropbox watcher share one event loop, so a per-run cap would still
# stack; a single module-level gate bounds the whole process. It also keeps the
# per-call wait_for budgets honest: without a gate, chunks queue in the default
# thread pool with their timeout clock already running. 16 = a 4h audio in one
# wave at 15-min chunks; the account's paid tier (1000+ RPM on pro) is nowhere
# near binding at this level.
MAX_CONCURRENT_GEMINI_CALLS = 16
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


# A pinned model id is not a guarantee: on 2026-10-08 Google deprecated
# gemini-3.7-flash and began answering every request for it with
# gemini-3.8-flash, with no error, and nobody noticed for three weeks because
# only the response's model_version showed it. Every generate_content response
# is therefore checked against the requested id.
# requested model id -> set of model_version values Google actually served.
SERVED_MODELS: dict[str, set[str]] = {}


def _check_served_model(requested: str, served: str | None) -> None:
    """Record which model answered, and warn once when a pinned id was rerouted.

    Aliases (`*-latest`) are expected to resolve to something else, so only a
    pinned id that comes back under a different name is reported. A version
    suffix on the same model ("gemini-x-001") is not a reroute.
    """
    if not served:
        return
    requested = requested.removeprefix("models/")
    served = served.removeprefix("models/")
    seen = SERVED_MODELS.setdefault(requested, set())
    if served in seen:
        return
    seen.add(served)
    if requested.endswith("-latest") or served == requested or served.startswith(requested + "-"):
        return
    logger.warning(
        f"Requested {requested} but Google served {served}: the pinned model is "
        "being rerouted, so results no longer reflect the model the pipeline was "
        "validated on"
    )


def served_model_mismatches() -> list[str]:
    """'requested -> served' lines for every pinned id that was rerouted."""
    out = []
    for requested, served in sorted(SERVED_MODELS.items()):
        if requested.endswith("-latest"):
            continue
        for s in sorted(served):
            if s != requested and not s.startswith(requested + "-"):
                out.append(f"{requested} -> {s}")
    return out
