"""G13-3: retry helper for transient chat failures.

Wraps chat_completion / chat_completion_stream calls with exponential backoff.
Only retries on transient errors (timeouts, connection errors). Does NOT retry
on 4xx (client error) or value errors (programming bug).

Usage:
    from qwable.fusion_retry import chat_with_retry

    response = chat_with_retry(
        client.chat_completion,
        max_retries=2,
        base_delay=1.0,
        model="m1",
        messages=[...],
    )
"""

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("qwable.fusion_retry")


# Exceptions worth retrying (transient). httpx errors + connection issues.
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if this exception is worth retrying."""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    # httpx errors (avoid hard dep on httpx here)
    name = type(exc).__name__
    if name in (
        "TimeoutException",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
    ):
        return True
    return False


def chat_with_retry(
    fn: Callable,
    *,
    max_retries: int = 2,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call fn(**kwargs) with exponential backoff on transient errors.

    Parameters:
        fn: callable (e.g. client.chat_completion)
        max_retries: number of retries AFTER the first attempt (0 = no retry)
        base_delay: initial backoff in seconds; doubles each retry
        **kwargs: passed to fn

    Returns:
        fn's return value on success
    Raises:
        Last exception if all retries exhausted
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(**kwargs)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "chat attempt %d/%d failed (%s: %s), retrying in %.1fs",
                attempt + 1, max_retries + 1,
                type(exc).__name__, exc, delay,
            )
            time.sleep(delay)
    # Unreachable, but mypy-friendly
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited without result")
