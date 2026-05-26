"""Centralized LLM provider with Gemini key rotation and Groq fallback.

Rotates through up to 4 Gemini API keys (env vars GEMINI_API_KEY,
GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4) using the
google.genai SDK. If all Gemini keys are exhausted, falls back to
Groq with a cascade of models.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import google.genai as genai
from google.genai import types as genai_types
from groq import AsyncGroq

from exceptions import LLMAuthError, LLMRateLimitError, LLMServiceError

__all__ = ["generate_text", "generate_text_with_tools"]

logger = logging.getLogger(__name__)

# ─── Key loading ──────────────────────────────────────────

_GEMINI_KEY_VARS = [
    "GEMINI_API_KEY",
    "GEMINI_API_KEY_2",
    "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_4",
]

GEMINI_MODEL = "gemini-2.5-flash"

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-4-scout",
    "llama-3.1-8b-instant",
]


def _get_gemini_keys() -> list[str]:
    """Collect non-empty Gemini keys from environment."""
    return [k for var in _GEMINI_KEY_VARS if (k := os.getenv(var, "").strip())]


def _is_retryable(e: Exception) -> bool:
    """Return True if the error suggests trying the next key/model."""
    s = str(e).lower()
    retryable = [
        "429", "quota", "rate limit", "resource_exhausted",
        "503", "overloaded", "unavailable", "500", "502",
    ]
    return any(k in s for k in retryable)


def _is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("401", "403", "invalid", "api_key", "permission"))


# ─── Gemini via google.genai ──────────────────────────────


async def _call_gemini(
    prompt: str,
    system_prompt: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    tools: list[Any] | None = None,
) -> str:
    """Try each Gemini key in rotation. Raises on total failure."""
    keys = _get_gemini_keys()
    if not keys:
        raise LLMServiceError("No Gemini API keys configured")

    last_exc: Exception | None = None
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
            )
            if tools:
                config.tools = tools

            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            if _is_auth_error(exc):
                logger.warning("Gemini auth error with key ...%s: %s", key[-4:], exc)
                continue
            if _is_retryable(exc):
                logger.warning("Gemini retryable error with key ...%s: %s", key[-4:], exc)
                continue
            raise LLMServiceError(str(exc)) from exc

    raise LLMRateLimitError(f"All Gemini keys exhausted: {last_exc}")


# ─── Groq fallback ────────────────────────────────────────


async def _call_groq(
    prompt: str,
    system_prompt: str | None,
    models: list[str],
    temperature: float,
    max_tokens: int,
    tools: list[Any] | None = None,
) -> str:
    """Try Groq models in order. Raises on total failure."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMServiceError("No GROQ_API_KEY configured")

    client = AsyncGroq(api_key=api_key)
    last_exc: Exception | None = None

    for model in models:
        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools

            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if _is_auth_error(exc):
                raise LLMAuthError(str(exc)) from exc
            if _is_retryable(exc):
                logger.warning("Groq retryable error on %s: %s", model, exc)
                continue
            logger.warning("Groq non-retryable error on %s: %s", model, exc)
            continue

    raise LLMServiceError(f"All Groq models failed: {last_exc}")


# ─── Public API ───────────────────────────────────────────


async def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str:
    """Generate text using Gemini (with key rotation) falling back to Groq.

    Args:
        prompt: The user prompt / input text.
        system_prompt: Optional system instruction.
        model: Override the default Gemini model name.
        temperature: Sampling temperature (0.0–2.0).
        max_tokens: Maximum output tokens.

    Returns:
        The generated text response.

    Raises:
        LLMAuthError: All credentials rejected.
        LLMRateLimitError: All keys rate-limited (before Groq fallback).
        LLMServiceError: Complete failure across all providers.
    """
    gemini_model = model or GEMINI_MODEL

    # Try Gemini first
    if _get_gemini_keys():
        try:
            return await _call_gemini(
                prompt, system_prompt, gemini_model, temperature, max_tokens
            )
        except (LLMRateLimitError, LLMServiceError) as exc:
            logger.info("Gemini failed, falling back to Groq: %s", exc)

    # Fallback to Groq
    return await _call_groq(prompt, system_prompt, GROQ_MODELS, temperature, max_tokens)


async def generate_text_with_tools(
    prompt: str,
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    model: str | None = None,
) -> str:
    """Generate text with tool/function-calling support.

    Uses the same Gemini → Groq fallback chain. Tools should be in the
    format expected by each provider's SDK (google.genai tool declarations
    for Gemini, OpenAI-style function schemas for Groq).

    Args:
        prompt: The user prompt.
        system_prompt: Optional system instruction.
        tools: Tool/function declarations for the model.
        model: Override the default Gemini model name.

    Returns:
        The generated text response.

    Raises:
        LLMAuthError: All credentials rejected.
        LLMServiceError: Complete failure across all providers.
    """
    gemini_model = model or GEMINI_MODEL

    if _get_gemini_keys():
        try:
            return await _call_gemini(
                prompt, system_prompt, gemini_model, 0.7, 8192, tools=tools
            )
        except (LLMRateLimitError, LLMServiceError) as exc:
            logger.info("Gemini (tools) failed, falling back to Groq: %s", exc)

    return await _call_groq(
        prompt, system_prompt, GROQ_MODELS, 0.7, 8192, tools=tools
    )
