"""Web search and API key management tools.

Provides DuckDuckGo (free) and Tavily (premium) web search with automatic
fallback, news-specific search, and a tool for persisting user-supplied
API keys to the JSON config store.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

from config import load_config, save_config, set_key

logger = logging.getLogger(__name__)

__all__ = ["build_search_tools"]

# ─── Key type mapping ─────────────────────────────────────────────────────────

_KEY_TYPE_MAP: dict[str, str] = {
    "gemini": "gemini_api_key",
    "anthropic": "anthropic_api_key",
    "github": "github_token",
    "tavily": "tavily_api_key",
    "huggingface": "huggingface_api_key",
    "grok": "grok_api_key",
    "groq": "groq_api_key",
    "openrouter": "openrouter_api_key",
    # Allow canonical names too
    "groq_api_key": "groq_api_key",
    "openrouter_api_key": "openrouter_api_key",
    "gemini_api_key": "gemini_api_key",
    "anthropic_api_key": "anthropic_api_key",
    "github_token": "github_token",
    "tavily_api_key": "tavily_api_key",
}

_FRIENDLY_NAMES: dict[str, str] = {
    "gemini_api_key": "Gemini API key",
    "anthropic_api_key": "Anthropic API key",
    "github_token": "GitHub token",
    "tavily_api_key": "Tavily API key",
    "huggingface_api_key": "HuggingFace API key",
    "grok_api_key": "Grok (xAI) API key",
    "groq_api_key": "Groq API key",
    "openrouter_api_key": "OpenRouter API key",
}


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _tavily_search(api_key: str, query: str, *, max_results: int = 6, topic: str = "general") -> str | None:
    """Run a Tavily search and return formatted results, or None on failure."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        depth = "advanced" if topic == "general" else "basic"
        results: dict[str, Any] = client.search(
            query, max_results=max_results, search_depth=depth, topic=topic
        )

        formatted: list[str] = []
        for r in results.get("results", []):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            if topic == "news":
                published = r.get("published_date", "")
                date_str = f" ({published})" if published else ""
                formatted.append(
                    f"📰 **{title}**{date_str}\n{content[:300]}…\n{url}"
                )
            else:
                score = r.get("score", 0)
                formatted.append(
                    f"**{title}**\nURL: {url}\nRelevance: {score:.2f}\n{content}"
                )

        answer = results.get("answer", "")
        output = ""
        if answer:
            output += f"**Direct answer:** {answer}\n\n"
        separator = "\n\n---\n\n" if topic == "general" else "\n\n"
        output += separator.join(formatted)
        return output or None
    except Exception as e:
        logger.warning("Tavily search failed (topic=%s): %s", topic, e)
        return None


def _ddg_web_search(query: str, *, max_results: int = 6) -> str:
    """Run a DuckDuckGo text search and return formatted results."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    lines = [
        f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}"
        for r in results
    ]
    return "\n\n---\n\n".join(lines)


def _ddg_news_search(query: str, *, max_results: int = 5) -> str:
    """Run a DuckDuckGo news search and return formatted results."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.news(query, max_results=max_results))
    if not results:
        return "No news found."
    lines: list[str] = []
    for r in results:
        date_str = f" ({r['date']})" if r.get("date") else ""
        lines.append(
            f"📰 **{r.get('title', '')}**{date_str}\n{r.get('body', '')[:300]}…\n{r.get('url', '')}"
        )
    return "\n\n".join(lines)


# ─── Public builder ───────────────────────────────────────────────────────────


def build_search_tools() -> list:
    """Build and return the list of search-related LangChain tools."""

    @tool
    def search_web(query: str) -> str:
        """Search the web for current, real-time information.

        Use for: news, events, prices, sports, weather, stocks, latest updates,
        anything that might have changed recently, or anything factual.
        Always cite sources. Summarize clearly.
        """
        cfg = load_config()
        tavily_key: str = cfg.get("tavily_api_key", "")

        if tavily_key:
            result = _tavily_search(tavily_key, query)
            if result:
                return result

        try:
            return _ddg_web_search(query)
        except Exception as e:
            logger.error("DuckDuckGo search failed: %s", e)
            return f"Search is temporarily unavailable. Error: {e}"

    @tool
    def search_news(query: str) -> str:
        """Search for the latest news on a topic.

        Returns recent news articles with headlines and summaries.
        Use specifically when user asks about news or current events.
        """
        cfg = load_config()
        tavily_key: str = cfg.get("tavily_api_key", "")

        if tavily_key:
            result = _tavily_search(
                tavily_key, f"latest news {query}", max_results=5, topic="news"
            )
            if result:
                return result

        try:
            return _ddg_news_search(query)
        except Exception as e:
            return f"News search unavailable: {e}"

    @tool
    def save_api_key(key_type: str, key_value: str) -> str:
        """Save an API key or token provided by the user.

        Args:
            key_type: Provider name (e.g. 'gemini', 'anthropic', 'github').
            key_value: The raw API key string.
        """
        actual_key = _KEY_TYPE_MAP.get(key_type.lower().strip())
        if not actual_key:
            return f"Unknown key type '{key_type}'."

        value = key_value.strip()

        # Gemini keys are pooled for rate-limit rotation
        if "gemini" in actual_key:
            cfg = load_config()
            keys: list[str] = cfg.get("gemini_api_keys", [])
            old: str = cfg.get("gemini_api_key", "")
            if old and old not in keys:
                keys.append(old)
            if value not in keys:
                keys.append(value)
            cfg["gemini_api_keys"] = keys
            cfg["gemini_api_key"] = value
            save_config(cfg)
            return (
                "✅ Gemini API key added to your pool of keys! "
                "I will cycle through them automatically to bypass rate limits."
            )

        set_key(actual_key, value)
        friendly = _FRIENDLY_NAMES.get(actual_key, actual_key)
        return f"✅ {friendly} saved successfully!"

    return [search_web, search_news, save_api_key]
