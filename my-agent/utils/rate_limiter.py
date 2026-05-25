import time
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter.
    Tracks per-user request counts within a time window.
    """

    def __init__(
        self,
        max_requests: int = 20,
        window_seconds: int = 60,
        cooldown_seconds: int = 30,
    ):
        self.max_requests = max_requests
        self.window = window_seconds
        self.cooldown = cooldown_seconds
        self._timestamps: dict[int, deque] = defaultdict(deque)
        self._blocked_until: dict[int, float] = {}

    def is_allowed(self, user_id: int) -> tuple[bool, float]:
        """
        Returns (allowed, wait_seconds).
        wait_seconds > 0 means user must wait that many seconds.
        """
        now = time.time()

        # Check if user is in cooldown
        blocked_until = self._blocked_until.get(user_id, 0)
        if now < blocked_until:
            return False, blocked_until - now

        # Clean old timestamps
        q = self._timestamps[user_id]
        while q and now - q[0] > self.window:
            q.popleft()

        if len(q) >= self.max_requests:
            # Apply cooldown
            self._blocked_until[user_id] = now + self.cooldown
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False, self.cooldown

        q.append(now)
        return True, 0

    def reset(self, user_id: int):
        self._timestamps[user_id].clear()
        self._blocked_until.pop(user_id, None)

    def get_remaining(self, user_id: int) -> int:
        now = time.time()
        q = self._timestamps[user_id]
        while q and now - q[0] > self.window:
            q.popleft()
        return max(0, self.max_requests - len(q))

    def get_stats(self, user_id: int) -> dict:
        now = time.time()
        q = self._timestamps.get(user_id, deque())
        active = [t for t in q if now - t <= self.window]
        blocked_until = self._blocked_until.get(user_id, 0)
        return {
            "requests_in_window": len(active),
            "remaining": max(0, self.max_requests - len(active)),
            "is_blocked": now < blocked_until,
            "unblock_in": max(0, blocked_until - now),
        }


class BurstRateLimiter:
    """Allows short bursts but throttles sustained high usage."""

    def __init__(self, burst_limit: int = 5, burst_window: int = 10, sustained_limit: int = 30, sustained_window: int = 300):
        self.burst = RateLimiter(burst_limit, burst_window, cooldown_seconds=15)
        self.sustained = RateLimiter(sustained_limit, sustained_window, cooldown_seconds=60)

    def is_allowed(self, user_id: int) -> tuple[bool, float, str]:
        burst_ok, burst_wait = self.burst.is_allowed(user_id)
        if not burst_ok:
            return False, burst_wait, f"Slow down! Wait {burst_wait:.0f}s"
        sustained_ok, sustained_wait = self.sustained.is_allowed(user_id)
        if not sustained_ok:
            return False, sustained_wait, f"Too many messages. Wait {sustained_wait:.0f}s"
        return True, 0, ""

    def reset(self, user_id: int):
        self.burst.reset(user_id)
        self.sustained.reset(user_id)


# Global singleton
global_rate_limiter = BurstRateLimiter(
    burst_limit=5,
    burst_window=10,
    sustained_limit=40,
    sustained_window=300,
)
