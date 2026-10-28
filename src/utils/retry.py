import asyncio
import logging
import random
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Cap the exponential backoff so a late attempt doesn't sleep for minutes.
MAX_RETRY_DELAY_MS = 60_000


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 6,
    base_delay_ms: int = 3000,
    context: str = "Operation",
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
) -> T:
    """
    Retry an async function with exponential backoff and jitter.

    The ladder is deliberately long. A newly released model can spend its first
    weeks returning 503 UNAVAILABLE under load: on 2026-08-14, four of five
    full-length transcriptions on gemini-3.7-flash died on the old 3-attempt/1s
    ladder while a single sequential run got through untouched. Retries cost
    nothing when the API is healthy.

    Args:
        fn: The async function to retry
        max_attempts: Maximum number of attempts (default: 6)
        base_delay_ms: Base delay in milliseconds (default: 3000)
        context: Context string for log messages (default: 'Operation')
        non_retryable_exceptions: Exception types that should propagate
            immediately without retrying (e.g. a content-policy block)

    Returns:
        The result of the function

    Raises:
        The original exception from the last attempt if all attempts fail, or
        immediately if a non-retryable exception is raised.
    """
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except non_retryable_exceptions:
            # Caller marked this class as not worth retrying; keep its type.
            raise
        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                # Exponential backoff, capped, with 0.5-1.5x jitter so a batch of
                # rate-limited calls doesn't retry in lockstep (thundering herd).
                delay = min(base_delay_ms * (2 ** (attempt - 1)), MAX_RETRY_DELAY_MS)
                delay *= 0.5 + random.random()
                logger.warning(
                    f"{context} failed (attempt {attempt}/{max_attempts}): {error}. "
                    f"Retrying in {int(delay)}ms..."
                )
                await asyncio.sleep(delay / 1000)

    # Re-raise the original exception so callers can branch on its type, rather
    # than a bare Exception that would erase it.
    assert last_error is not None
    raise last_error


class ResponseBlockedError(Exception):
    """Gemini returned no text, with prompt_feedback.block_reason set (OTHER).

    It belongs to the (model, request) pair and is slow to find out: in the
    transcribe skill, all 23 gemini-3.8-flash requests for one 14-minute chunk
    of an October 2026 call made with four of the five prompt variants tried
    came back blocked, uploaded or inlined, each after up to five minutes of
    thinking, while gemini-3.1-pro and Argon transcribed it (in older runs, a
    retry of a blocked request got through once in 35). So with_model_fallback
    switches models on the first one.
    """


# A model whose capacity has collapsed cannot be retried out of: on 2026-08-14
# gemini-3.7-flash answered 2 of 16 calls, the rest 503 UNAVAILABLE, while
# gemini-3.1-pro answered 10 of 10 in the same minute and at the same
# concurrency. Serial calls fared no better than parallel ones, so no backoff
# ladder reaches the other side of it.
FALLBACK_MODEL = "gemini-pro-latest"


async def with_model_fallback(
    make_call: Callable[[str], Callable[[], Awaitable[T]]],
    model: str,
    context: str = "Operation",
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
    fallback_model: str = FALLBACK_MODEL,
) -> T:
    """Run `make_call(model)` through with_retry, then retry on another model.

    `make_call` takes a model id and returns the zero-argument coroutine
    function that with_retry expects, so the fallback attempt rebuilds the
    request against a different model rather than replaying the first one.

    Exhausting the retry ladder triggers the switch, and so does the first
    ResponseBlockedError: that block belongs to the (model, request) pair, so
    retrying the same model only repeats it, minutes at a time. Any other
    non-retryable failure is a property of the request alone (a malformed
    prompt) and would fail on any model, so it propagates untouched.
    """
    stop = non_retryable_exceptions + (ResponseBlockedError,)
    try:
        return await with_retry(
            make_call(model), context=f"{context} on {model}",
            non_retryable_exceptions=stop,
        )
    except non_retryable_exceptions:
        raise
    except Exception as error:
        if model == fallback_model:
            raise
        if isinstance(error, ResponseBlockedError):
            logger.warning(f"{context}: {error}; switching to {fallback_model} without retrying")
        else:
            logger.warning(
                f"{context}: {model} exhausted its retries ({error}); "
                f"falling back to {fallback_model} for this call"
            )
        return await with_retry(
            make_call(fallback_model),
            context=f"{context} on {fallback_model}",
            non_retryable_exceptions=stop,
        )
