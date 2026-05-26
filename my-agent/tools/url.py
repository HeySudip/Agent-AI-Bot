"""URL fetching, summarization, and link extraction tools.

All outbound requests pass through :func:`safety.ssrf_guard.assert_url_is_safe`
at every redirect hop to prevent SSRF attacks against internal services,
cloud-metadata endpoints, and private networks.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import requests
from langchain.tools import tool

from safety.ssrf_guard import SSRFBlockedError, assert_url_is_safe

logger = logging.getLogger(__name__)

__all__ = ["build_url_tools", "fetch_url_content"]

# ─── Constants ────────────────────────────────────────────────────────────────

_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_MAX_REDIRECTS = 5

# ─── SSRF-safe HTTP client ────────────────────────────────────────────────────


def _safe_get(url: str, *, timeout: int = 20, allow_redirects: bool = True) -> requests.Response:
    """Issue a GET request that validates SSRF safety at every redirect hop.

    Args:
        url: Target URL (must be http or https).
        timeout: Request timeout in seconds.
        allow_redirects: If False, return the first response without following redirects.

    Returns:
        The final :class:`requests.Response`.

    Raises:
        SSRFBlockedError: If any hop resolves to a blocked address.
        requests.exceptions.TooManyRedirects: If redirect chain exceeds limit.
    """
    assert_url_is_safe(url)

    if not allow_redirects:
        return requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=False)

    session = requests.Session()
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        resp = session.get(current, headers=_HEADERS, timeout=timeout, allow_redirects=False)
        if not (resp.is_redirect or resp.is_permanent_redirect):
            return resp
        next_url = resp.headers.get("location", "")
        if not next_url:
            return resp
        if next_url.startswith("/"):
            next_url = urljoin(current, next_url)
        assert_url_is_safe(next_url)
        current = next_url

    raise requests.exceptions.TooManyRedirects(f"More than {_MAX_REDIRECTS} redirects.")


# ─── Content extraction ───────────────────────────────────────────────────────


def fetch_url_content(url: str, max_chars: int = 10000) -> str:
    """Fetch and extract readable text from a URL.

    Handles HTML (with article/main content extraction), JSON, and plain text.
    Returns an error string (prefixed with "Error:") on failure rather than raising.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return.

    Returns:
        Extracted text content or an error message string.
    """
    try:
        try:
            resp = _safe_get(url, timeout=20, allow_redirects=True)
        except SSRFBlockedError as exc:
            return f"Error: refused to fetch this URL — {exc}"

        resp.raise_for_status()
        content_type: str = resp.headers.get("content-type", "")

        if "application/json" in content_type:
            return resp.text[:max_chars]

        if "text/plain" in content_type:
            return resp.text[:max_chars]

        # HTML extraction
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.content, "html.parser")

        # Try semantic content containers first
        for selector in ["article", "main", '[role="main"]', ".content", "#content", ".post-content"]:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return _clean_text(text)[:max_chars]

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "advertisement", ".ads", "#ads", ".cookie", ".popup"]):
            tag.decompose()

        # Extract paragraphs and headings
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
        text = "\n".join(
            p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30
        )

        if not text:
            text = soup.get_text(separator="\n", strip=True)

        return _clean_text(text)[:max_chars]

    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} — {e}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the URL."
    except Exception as e:
        return f"Error fetching URL: {e}"


def _clean_text(text: str) -> str:
    """Remove excessive whitespace, blank lines, and duplicate consecutive lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned: list[str] = []
    prev: str | None = None
    for line in lines:
        if line != prev:
            cleaned.append(line)
            prev = line
    return "\n".join(cleaned)


# ─── URL classification helpers ───────────────────────────────────────────────


def is_github_url(url: str) -> bool:
    """Return True if the URL points to github.com."""
    return "github.com" in url


def is_youtube_url(url: str) -> bool:
    """Return True if the URL points to YouTube."""
    return any(d in url for d in ("youtube.com", "youtu.be"))


def extract_github_info(url: str) -> dict[str, str | bool]:
    """Parse a github.com URL to extract owner and repo.

    Returns:
        Dict with 'owner', 'repo', and 'matched' keys.
    """
    patterns = [
        r"github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)",
        r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)",
        r"github\.com/([^/]+)/([^/]+)/?$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return {"owner": m.group(1), "repo": m.group(2), "matched": True}
    return {"matched": False}


# ─── Public builder ───────────────────────────────────────────────────────────


def build_url_tools() -> list:
    """Build and return the list of URL-related LangChain tools."""

    @tool
    def fetch_and_summarize_url(url: str) -> str:
        """Fetch the full content of any URL or web article so you can read and summarize it.

        Use when the user pastes a link and wants a summary, analysis, or has
        questions about it. Works for articles, blog posts, documentation,
        GitHub pages, etc.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Special handling for raw GitHub file content
        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            content = fetch_url_content(raw_url, max_chars=8000)
            if not content.startswith("Error"):
                return f"**GitHub file content:**\n\n```\n{content}\n```"

        content = fetch_url_content(url)
        if content.startswith("Error"):
            return content

        word_count = len(content.split())
        return f"**Fetched content** ({word_count} words) from {url}:\n\n{content}"

    @tool
    def get_page_title_and_description(url: str) -> str:
        """Get the title and meta description of a web page quickly.

        Useful for checking what a link is about before doing a full fetch.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            try:
                resp = _safe_get(url, timeout=10, allow_redirects=True)
            except SSRFBlockedError as exc:
                return f"Error: refused to fetch this URL — {exc}"
            resp.raise_for_status()

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.content, "html.parser")

            title_tag = soup.find("title")
            title_text = title_tag.get_text(strip=True) if title_tag else "No title found"

            desc_tag = (
                soup.find("meta", attrs={"name": "description"})
                or soup.find("meta", attrs={"property": "og:description"})
            )
            desc_text = desc_tag.get("content", "No description found") if desc_tag else "No description"

            og_title_tag = soup.find("meta", attrs={"property": "og:title"})
            og_title_text = og_title_tag.get("content", "") if og_title_tag else ""

            return (
                f"**Title:** {og_title_text or title_text}\n"
                f"**Description:** {desc_text}\n"
                f"**URL:** {url}"
            )
        except Exception as e:
            return f"Error: {e}"

    @tool
    def extract_links_from_url(url: str) -> str:
        """Extract all links from a web page.

        Useful for finding references, documentation links, or related pages.
        Returns up to 30 external links.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            try:
                resp = _safe_get(url, timeout=15, allow_redirects=True)
            except SSRFBlockedError as exc:
                return f"Error: refused to fetch this URL — {exc}"
            resp.raise_for_status()

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.content, "html.parser")
            links: list[str] = []
            seen: set[str] = set()
            for a in soup.find_all("a", href=True):
                href: str = a["href"]
                text: str = a.get_text(strip=True)
                if href.startswith("http") and href not in seen and text:
                    links.append(f"• [{text[:80]}]({href})")
                    seen.add(href)
                if len(links) >= 30:
                    break
            return "\n".join(links) if links else "No external links found on this page."
        except Exception as e:
            return f"Error: {e}"

    return [fetch_and_summarize_url, get_page_title_and_description, extract_links_from_url]
