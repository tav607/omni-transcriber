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
