"""Tests for the burst + sustained rate limiter."""

from __future__ import annotations

import pytest

from utils.rate_limiter import BurstRateLimiter, RateLimiter


class TestRateLimiter:
    def test_allows_under_limit(self) -> None:
        rl = RateLimiter(max_requests=3, window_seconds=60, cooldown_seconds=10)
        for _ in range(3):
            allowed, wait = rl.is_allowed(user_id=1)
            assert allowed is True
            assert wait == 0

    def test_blocks_over_limit(self) -> None:
        rl = RateLimiter(max_requests=2, window_seconds=60, cooldown_seconds=15)
        rl.is_allowed(1)
        rl.is_allowed(1)
        allowed, wait = rl.is_allowed(1)
        assert allowed is False
        assert wait == pytest.approx(15, abs=1)

    def test_reset_clears_state(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60, cooldown_seconds=15)
        rl.is_allowed(1)
        rl.is_allowed(1)  # blocks
        rl.reset(1)
        allowed, _ = rl.is_allowed(1)
        assert allowed is True

    def test_users_isolated(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60, cooldown_seconds=10)
        assert rl.is_allowed(1)[0] is True
        assert rl.is_allowed(2)[0] is True

    def test_get_remaining(self) -> None:
        rl = RateLimiter(max_requests=5, window_seconds=60, cooldown_seconds=10)
        assert rl.get_remaining(1) == 5
        rl.is_allowed(1)
        rl.is_allowed(1)
        assert rl.get_remaining(1) == 3


class TestBurstRateLimiter:
    def test_short_burst_allowed(self) -> None:
        rl = BurstRateLimiter(burst_limit=3, burst_window=10, sustained_limit=100, sustained_window=600)
        for _ in range(3):
            allowed, _, _ = rl.is_allowed(1)
            assert allowed is True

    def test_burst_excess_blocked(self) -> None:
        rl = BurstRateLimiter(burst_limit=2, burst_window=10, sustained_limit=100, sustained_window=600)
        rl.is_allowed(1)
        rl.is_allowed(1)
        allowed, wait, reason = rl.is_allowed(1)
        assert allowed is False
        assert wait > 0
        assert "Slow down" in reason

    def test_reset_unblocks(self) -> None:
        rl = BurstRateLimiter(burst_limit=1, burst_window=10, sustained_limit=100, sustained_window=600)
        rl.is_allowed(1)
        rl.is_allowed(1)
        rl.reset(1)
        allowed, _, _ = rl.is_allowed(1)
        assert allowed is True
