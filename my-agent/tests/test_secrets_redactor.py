"""Tests for the secrets redactor."""

from __future__ import annotations

import pytest

from safety.secrets_redactor import (
    REDACTION_PLACEHOLDER,
    redact_secrets,
    structlog_redactor,
)


class TestRedactStrings:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",
            "sk-or-v1-abcdefghijklmnopqrstuvwxyz1234567890",
            "gsk_abcdefghijklmnopqrstuvwxyz1234567890",
            "xai-abcdefghijklmnopqrstuvwxyz1234567890",
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
            "github_pat_" + "A" * 70,
            "xoxb-12345-67890-abcdefghijklmnop",
            "tvly-abcdefghijklmnopqrstuvwxyz12",
            "hf_abcdefghijklmnopqrstuvwxyz1234567890",
            "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567",
            "AKIAIOSFODNN7EXAMPLE",
            "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY.SflKxwRJSMeKKF",
            "1234567890:ABCdefGHIjklMNOpqrSTUvwxyz1234567890",
        ],
    )
    def test_secret_is_redacted(self, secret: str) -> None:
        message = f"prefix {secret} suffix"
        out = redact_secrets(message)
        assert secret not in out
        assert REDACTION_PLACEHOLDER in out

    def test_plain_text_unchanged(self) -> None:
        assert redact_secrets("Hello, world!") == "Hello, world!"

    def test_empty_input(self) -> None:
        assert redact_secrets("") == ""

    def test_non_string_returned_as_is(self) -> None:
        assert redact_secrets(None) is None  # type: ignore[arg-type]
        assert redact_secrets(42) == 42  # type: ignore[arg-type]


class TestStructlogProcessor:
    def test_redacts_string_values(self) -> None:
        event = {"event": "login", "msg": "key=ghp_" + "A" * 36}
        out = structlog_redactor(None, "info", event)
        assert "ghp_" not in out["msg"]

    def test_redacts_sensitive_key_names(self) -> None:
        event = {"event": "x", "api_key": "anything-here", "password": "hunter2"}
        out = structlog_redactor(None, "info", event)
        assert out["api_key"] == REDACTION_PLACEHOLDER
        assert out["password"] == REDACTION_PLACEHOLDER

    def test_handles_nested_dicts(self) -> None:
        event = {
            "event": "x",
            "ctx": {"token": "ghp_" + "B" * 36, "user": "alice"},
        }
        out = structlog_redactor(None, "info", event)
        assert out["ctx"]["token"] == REDACTION_PLACEHOLDER
        assert out["ctx"]["user"] == "alice"

    def test_handles_lists(self) -> None:
        event = {"event": "x", "items": ["normal", "AIzaSy" + "C" * 35]}
        out = structlog_redactor(None, "info", event)
        assert "AIzaSy" not in out["items"][1]
