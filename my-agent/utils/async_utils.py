"""Small helpers for async code paths: timeouts and retry policy."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

__all__ = ["with_timeout", "retry_async"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_timeout(
    coro: Awaitable[T],
    seconds: float,
    *,
    label: str = "operation",
) -> T:
    """Run *coro* with a deadline.

    Args:
        coro: The awaitable to execute.
        seconds: Maximum wall-clock time allowed.
        label: Human-readable name for log messages.

    Raises:
        asyncio.TimeoutError: If the deadline is exceeded.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning("Timeout after %.2fs in %s", seconds, label)
        raise


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    label: str = "retry",
) -> T:
    """Call *fn* up to *attempts* times with exponential backoff + jitter.

    Re-raises the last exception if all attempts fail. Only retries when the
    exception is an instance of one of *retry_on*.

    Args:
        fn: Zero-argument async callable to invoke.
        attempts: Maximum number of tries (must be >= 1).
        base_delay_s: Initial backoff delay in seconds.
        max_delay_s: Cap on the computed delay.
        retry_on: Exception types that trigger a retry.
        label: Human-readable name for log messages.

    Raises:
        ValueError: If *attempts* < 1.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == attempts:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()
            logger.info(
                "Retrying %s after error (%s), attempt %d/%d in %.2fs",
                label,
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None  # noqa: S101 — guaranteed by loop logic
    raise last_exc
