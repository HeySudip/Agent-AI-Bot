"""Structured logging configuration with secret redaction.

Falls back gracefully to stdlib logging if structlog is not installed, so
existing imports keep working in environments that have not yet upgraded.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from safety.secrets_redactor import structlog_redactor

__all__ = ["configure_logging", "get_logger"]


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    resolved = logging.getLevelName(env_level)
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(
    level: str | int | None = None,
    *,
    fmt: str | None = None,
) -> None:
    """Configure the root logger and structlog (if available).

    Args:
        level: Log level name or int. Defaults to ``LOG_LEVEL`` env var, then INFO.
        fmt: ``"json"`` or ``"console"``. Defaults to ``LOG_FORMAT`` env var, then
            ``"console"`` for interactive terminals and ``"json"`` otherwise.
    """
    log_level = _resolve_level(level)
    log_format = (fmt or os.getenv("LOG_FORMAT") or "").lower()
    if log_format not in {"json", "console"}:
        log_format = "console" if sys.stdout.isatty() else "json"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quiet down chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "telegram", "urllib3"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))

    try:
        import structlog
    except ImportError:  # pragma: no cover - optional dependency
        return

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog_redactor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger if available, else a stdlib logger."""
    try:
        import structlog

        return structlog.get_logger(name) if name else structlog.get_logger()
    except ImportError:  # pragma: no cover
        return logging.getLogger(name or __name__)
