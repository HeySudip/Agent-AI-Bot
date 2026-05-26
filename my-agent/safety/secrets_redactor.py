"""Redact common secret patterns from text before logging.

Designed to run as a structlog processor on every log event. The patterns
target high-confidence secret formats (provider-specific prefixes plus a
length floor) to keep false positives low. We do NOT try to catch every
possible secret — the goal is defense-in-depth against accidental leaks of
the most common ones.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from re import Pattern
from typing import Any, Final

__all__ = ["REDACTION_PLACEHOLDER", "redact_secrets", "structlog_redactor"]

REDACTION_PLACEHOLDER: Final[str] = "[REDACTED]"

_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    # OpenAI keys
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    # Anthropic keys
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    # OpenRouter keys
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}\b"),
    # Groq keys
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    # xAI / Grok
    re.compile(r"\bxai-[A-Za-z0-9]{20,}\b"),
    # GitHub personal access tokens (classic + fine-grained)
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bghu_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bghs_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgho_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bghr_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    # Slack tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    # Tavily
    re.compile(r"\btvly-[A-Za-z0-9_-]{20,}\b"),
    # HuggingFace
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    # Google AI Studio / Gemini
    re.compile(r"\bAIzaSy[A-Za-z0-9_-]{30,}\b"),
    # AWS access key ID
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    # JSON web tokens (three base64url segments)
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    # Telegram bot tokens (numeric:35-char-secret)
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
)

_REDACT_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "auth",
)


def redact_secrets(text: str) -> str:
    """Return *text* with known secret patterns replaced by the placeholder."""
    if not isinstance(text, str) or not text:
        return text
    redacted = text
    for pattern in _PATTERNS:
        redacted = pattern.sub(REDACTION_PLACEHOLDER, redacted)
    return redacted


def _key_looks_sensitive(key: str) -> bool:
    """Return True if *key* looks like it holds a secret value."""
    lowered = key.lower()
    return any(fragment in lowered for fragment in _REDACT_KEY_FRAGMENTS)


def _redact_value(value: Any) -> Any:
    """Recursively redact secrets within a value."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: _redact_event_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    return value


def _redact_event_value(key: str, value: Any) -> Any:
    """Redact a value, applying full redaction if the key looks sensitive."""
    if isinstance(value, str) and _key_looks_sensitive(key):
        return REDACTION_PLACEHOLDER if value else value
    return _redact_value(value)


def structlog_redactor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: redact secrets in every value of the event dict."""
    return {k: _redact_event_value(k, v) for k, v in event_dict.items()}


def iter_known_patterns() -> Iterable[Pattern[str]]:
    """Expose the compiled patterns for tests and diagnostics."""
    return _PATTERNS
