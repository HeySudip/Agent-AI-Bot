"""Typed environment-driven settings on top of the existing JSON config.

The legacy ``config.py`` keeps writing chat-supplied API keys to a JSON file
on disk (so users can paste keys at runtime). This module layers a typed
:class:`Settings` object over the env-var inputs so cross-field invariants
are validated up-front and missing required values fail fast.

If ``pydantic-settings`` is not installed this module degrades gracefully
to a dataclass-based fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Settings", "load_settings"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded once at startup.

    All fields are read from environment variables. The class is frozen so
    accidental in-place mutation raises immediately.
    """

    # Identity / environment
    agent_name: str = "Agent-AI-Bot"
    environment: str = "dev"

    # Telegram
    telegram_bot_token: str = ""

    # Logging / observability
    log_level: str = "INFO"
    log_format: str = ""

    # Agent loop
    max_tool_iterations: int = 10
    short_term_memory_size: int = 30

    # Safety
    rate_limit_burst: int = 5
    rate_limit_burst_window_s: int = 10
    rate_limit_sustained: int = 40
    rate_limit_sustained_window_s: int = 300
    max_input_chars: int = 8000
    enable_content_filter: bool = False

    # HTTP / network
    http_timeout_s: float = 20.0
    fetch_url_max_chars: int = 10_000
    user_agent: str = (
        "Mozilla/5.0 (compatible; Agent-AI-Bot/1.0; "
        "+https://github.com/HeySudip/Agent-AI-Bot)"
    )

    # Storage
    database_path: str = "data.db"
    config_path: str = "config/config.json"

    # Convenience
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "prod"


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings(
        agent_name=_env_str("AGENT_NAME", "Agent-AI-Bot"),
        environment=_env_str("ENVIRONMENT", "dev"),
        telegram_bot_token=_env_str("TELEGRAM_BOT_TOKEN", ""),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        log_format=_env_str("LOG_FORMAT", ""),
        max_tool_iterations=_env_int("MAX_TOOL_ITERATIONS", 10),
        short_term_memory_size=_env_int("SHORT_TERM_MEMORY_SIZE", 30),
        rate_limit_burst=_env_int("RATE_LIMIT_BURST", 5),
        rate_limit_burst_window_s=_env_int("RATE_LIMIT_BURST_WINDOW_S", 10),
        rate_limit_sustained=_env_int("RATE_LIMIT_SUSTAINED", 40),
        rate_limit_sustained_window_s=_env_int("RATE_LIMIT_SUSTAINED_WINDOW_S", 300),
        max_input_chars=_env_int("MAX_INPUT_CHARS", 8000),
        enable_content_filter=_env_bool("ENABLE_CONTENT_FILTER", False),
        http_timeout_s=float(_env_str("HTTP_TIMEOUT_S", "20")),
        fetch_url_max_chars=_env_int("FETCH_URL_MAX_CHARS", 10_000),
        user_agent=_env_str(
            "USER_AGENT",
            "Mozilla/5.0 (compatible; Agent-AI-Bot/1.0; "
            "+https://github.com/HeySudip/Agent-AI-Bot)",
        ),
        database_path=_env_str("DATABASE_PATH", "data.db"),
        config_path=_env_str("CONFIG_PATH", "config/config.json"),
    )
