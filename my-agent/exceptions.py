"""Structured exception hierarchy for the agent.

Catching specific subclasses lets us map internal failure modes to friendly
user-facing messages and decide which errors are worth retrying.
"""

from __future__ import annotations

__all__ = [
    "AgentError",
    "ConfigError",
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMServiceError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "MemoryError",
    "SafetyError",
    "RateLimitedError",
    "ContentBlockedError",
]


class AgentError(Exception):
    """Base class for every error raised by the agent itself."""


class ConfigError(AgentError):
    """Raised when configuration is missing or invalid."""


class LLMError(AgentError):
    """Base for all LLM-call failures."""


class LLMAuthError(LLMError):
    """The provider rejected the credentials."""


class LLMRateLimitError(LLMError):
    """The provider returned a 429 / quota exhausted response."""


class LLMTimeoutError(LLMError):
    """The LLM call exceeded its deadline."""


class LLMServiceError(LLMError):
    """The provider returned a transient 5xx or unavailable status."""


class ToolError(AgentError):
    """Base for all tool-related failures."""


class ToolNotFoundError(ToolError):
    """The agent asked for a tool that is not registered."""


class ToolValidationError(ToolError):
    """Tool input failed schema validation."""


class ToolExecutionError(ToolError):
    """Tool raised while executing."""


class MemoryError(AgentError):  # noqa: N818 — intentional name shadowing builtin
    """Raised when the memory subsystem cannot be reached."""


class SafetyError(AgentError):
    """Base for safety / policy errors surfaced to the user."""


class RateLimitedError(SafetyError):
    """The user exceeded the per-user rate limit."""

    def __init__(self, retry_after_s: float, message: str | None = None) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(message or f"Rate limited; retry in {retry_after_s:.1f}s.")


class ContentBlockedError(SafetyError):
    """Input or output was rejected by the content filter."""
