"""Shared Gemini plumbing: the Vertex AI client, model paths, response checks."""

import asyncio
import functools
import logging
import os
from pathlib import Path

from google import genai

logger = logging.getLogger(__name__)

# Gemini runs on Vertex AI, which bills through Cloud Billing (postpay); AI
# Studio moved to prepaid credit in 2026. Credentials are a Vertex AI API key
# and the GCP project id it belongs to: one bare value per file, shared with
# the other Gemini jobs on this machine, or the two env vars (which win; the
# offline tests use them).
VERTEX_KEY_FILE = Path("~/.vertex-key").expanduser()
VERTEX_PROJECT_FILE = Path("~/.vertex-project").expanduser()
VERTEX_LOCATION = "global"


def _read_bare(path: Path) -> str:
    return path.read_text().strip() if path.exists() else ""


@functools.lru_cache(maxsize=None)
def vertex_credentials() -> tuple[str, str]:
    """(API key, project id); either is "" when it is configured nowhere."""
    key = os.getenv("VERTEX_API_KEY", "").strip() or _read_bare(VERTEX_KEY_FILE)
    project = os.getenv("VERTEX_PROJECT", "").strip() or _read_bare(VERTEX_PROJECT_FILE)
    return key, project


# Module-level client cache: reuse genai.Client to avoid recreating HTTP connection pools
_client_cache: dict[str, genai.Client] = {}


def get_client(api_key: str) -> genai.Client:
    """Get or create the shared Vertex AI client for the given API key.

    base_url is left unset so the google-genai SDK's own GOOGLE_VERTEX_BASE_URL
    environment variable takes effect when Gemini has to go through a relay.
    """
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(vertexai=True, api_key=api_key)
    return _client_cache[api_key]


def model_path(model: str) -> str:
    """Full resource name of `model` on the global endpoint.

    A key alone gets requests routed to us-central1, which does not serve
    gemini-4-argon; naming the project and the global location does. An id
    that is already a path passes through.
    """
    project = vertex_credentials()[1]
    if not project or "/" in model:
        return model
    return f"projects/{project}/locations/{VERTEX_LOCATION}/publishers/google/models/{model}"

# Cap total in-flight Gemini generate_content calls across every fan-out
# (transcription chunk waves, editor edit/podcast-edit/translation). The
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
