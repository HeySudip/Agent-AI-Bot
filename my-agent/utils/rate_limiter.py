"""Sliding-window rate limiter with burst and sustained limits."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Final

__all__ = ["RateLimiter", "BurstRateLimiter", "global_rate_limiter"]

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window rate limiter with per-user cooldown on breach."""

    def __init__(
        self,
        max_requests: int = 20,
        window_seconds: int = 60,
        cooldown_seconds: int = 30,
    ) -> None:
        self.max_requests: Final = max_requests
        self.window: Final = window_seconds
        self.cooldown: Final = cooldown_seconds
        self._timestamps: dict[int, deque[float]] = defaultdict(deque)
        self._blocked_until: dict[int, float] = {}

    def is_allowed(self, user_id: int) -> tuple[bool, float]:
        """Check whether *user_id* may proceed.

        Returns:
            A tuple ``(allowed, wait_seconds)``. If not allowed,
            *wait_seconds* indicates how long the user must wait.
        """
        now = time.time()

        blocked_until = self._blocked_until.get(user_id, 0.0)
        if now < blocked_until:
            return False, blocked_until - now

        # Evict expired timestamps
        q = self._timestamps[user_id]
        while q and now - q[0] > self.window:
            q.popleft()

        if len(q) >= self.max_requests:
            self._blocked_until[user_id] = now + self.cooldown
            logger.warning("Rate limit exceeded for user %d", user_id)
            return False, float(self.cooldown)

        q.append(now)
        return True, 0.0

    def reset(self, user_id: int) -> None:
        """Clear all state for *user_id*."""
        self._timestamps[user_id].clear()
        self._blocked_until.pop(user_id, None)

    def get_remaining(self, user_id: int) -> int:
        """Return how many requests *user_id* has left in the current window."""
        now = time.time()
        q = self._timestamps[user_id]
        while q and now - q[0] > self.window:
            q.popleft()
        return max(0, self.max_requests - len(q))

    def get_stats(self, user_id: int) -> dict[str, int | float | bool]:
        """Return diagnostic info for *user_id*."""
        now = time.time()
        q = self._timestamps.get(user_id, deque())
        active = [t for t in q if now - t <= self.window]
        blocked_until = self._blocked_until.get(user_id, 0.0)
        return {
            "requests_in_window": len(active),
            "remaining": max(0, self.max_requests - len(active)),
            "is_blocked": now < blocked_until,
            "unblock_in": max(0.0, blocked_until - now),
        }


class BurstRateLimiter:
    """Composite limiter: allows short bursts but throttles sustained usage."""

    def __init__(
        self,
        burst_limit: int = 5,
        burst_window: int = 10,
        sustained_limit: int = 30,
        sustained_window: int = 300,
    ) -> None:
        self.burst = RateLimiter(burst_limit, burst_window, cooldown_seconds=15)
        self.sustained = RateLimiter(
            sustained_limit, sustained_window, cooldown_seconds=60
        )

    def is_allowed(self, user_id: int) -> tuple[bool, float, str]:
        """Check both burst and sustained limits.

        Returns:
            A tuple ``(allowed, wait_seconds, message)``.
        """
        burst_ok, burst_wait = self.burst.is_allowed(user_id)
        if not burst_ok:
            return False, burst_wait, f"Slow down! Wait {burst_wait:.0f}s"
        sustained_ok, sustained_wait = self.sustained.is_allowed(user_id)
        if not sustained_ok:
            return False, sustained_wait, f"Too many messages. Wait {sustained_wait:.0f}s"
        return True, 0.0, ""

    def reset(self, user_id: int) -> None:
        """Reset both burst and sustained state for *user_id*."""
        self.burst.reset(user_id)
        self.sustained.reset(user_id)


# Global singleton used by handlers
global_rate_limiter = BurstRateLimiter(
    burst_limit=5,
    burst_window=10,
    sustained_limit=40,
    sustained_window=300,
)
