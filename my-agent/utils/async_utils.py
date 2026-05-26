"""Small helpers for async code paths: timeouts and retry policy."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["with_timeout", "retry_async"]


async def with_timeout(
    coro: Awaitable[T],
    seconds: float,
    *,
    label: str = "operation",
) -> T:
    """Run ``coro`` with a deadline. Raise :class:`asyncio.TimeoutError`."""
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
    """Call ``fn`` up to ``attempts`` times with exponential backoff + jitter.

    Re-raises the last exception if all attempts fail. Only retries when the
    exception is an instance of one of ``retry_on``.
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
            delay = delay * (0.5 + random.random())
            logger.info(
                "Retrying %s after error (%s), attempt %d/%d in %.2fs",
                label,
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # nosec - by construction above
    raise last_exc
