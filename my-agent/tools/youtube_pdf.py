"""Document-style PDF tools (research and free-form text → PDF).

Video-related work has moved to :mod:`tools.video`. This module contains
the general-purpose research and text-to-PDF tools.

The legacy ``extract_youtube_to_pdf`` / ``search_and_extract_youtube_to_pdf``
/ ``youtube_video_to_pdf`` symbols are kept as thin shims that forward to
the new video tool, so any saved chat prompts referencing those names still work.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from langchain.tools import tool

from safety.ssrf_guard import SSRFBlockedError, assert_url_is_safe

from .pdf_builder import (
    Bullets,
    Heading,
    PdfBuildError,
    PdfMeta,
    Rule,
    build_pdf,
)
from .pdf_builder import Paragraph as PdfParagraph
from .video import run_video_to_pdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _llm_compile(prompt: str, *, max_tokens: int = 4000, temperature: float = 0.2) -> str:
    """Call llm_provider.generate_text synchronously with fallback handling.

    Uses the centralized provider (Gemini key rotation → Groq fallback).
    Returns the generated text, or empty string on failure.
    """
    try:
        from llm_provider import generate_text
    except ImportError as exc:
        logger.warning("llm_provider not available: %s", exc)
        return ""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                text = pool.submit(
                    asyncio.run,
                    generate_text(prompt, temperature=temperature, max_tokens=max_tokens),
                ).result(timeout=60)
        else:
            text = asyncio.run(
                generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
            )
    except Exception as exc:
        logger.warning("LLM compilation failed: %s", exc)
        return ""

    return text.strip() if text else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(seed: str, ext: str = ".pdf") -> str:
    """Generate a safe temporary file path from a seed string."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", seed).strip("_")[:60] or "document"
    return f"/tmp/{safe}_{int(time.time())}{ext}"


def _ddg_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """DuckDuckGo text search with a Bing scrape fallback."""
    for pkg in ("ddgs", "duckduckgo_search"):
        try:
            if pkg == "ddgs":
                from ddgs import DDGS  # type: ignore[import-untyped]
            else:
                from duckduckgo_search import DDGS  # type: ignore[import-untyped]
            with DDGS() as d:
                results = list(d.text(query, max_results=max_results))
            if results:
                return results
        except Exception as exc:
            logger.debug("Search via %s failed: %s", pkg, exc)

    # Bing HTML fallback
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count={max_results}"
        try:
            assert_url_is_safe(url)
        except SSRFBlockedError as exc:
            logger.warning("Bing fallback URL refused: %s", exc)
            return []
        r = requests.get(url, headers=headers, timeout=10)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, "lxml")
        results: list[dict[str, Any]] = []
        for li in soup.select("li.b_algo")[:max_results]:
            anchor = li.select_one("h2 a")
            snippet = li.select_one(".b_caption p")
            if anchor:
                results.append(
                    {
                        "title": anchor.get_text(strip=True),
                        "href": anchor.get("href", ""),
                        "body": snippet.get_text(strip=True) if snippet else "",
                    }
                )
        return results
    except Exception as exc:
        logger.warning("Bing fallback search failed: %s", exc)
        return []


def _scrape_url(url: str, timeout: int = 10) -> str:
    """Scrape readable text from a public URL with several UA fallbacks."""
    try:
        assert_url_is_safe(url)
    except SSRFBlockedError as exc:
        logger.info("Refusing to scrape unsafe URL %s: %s", url, exc)
        return ""

    user_agents = [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ),
        "Googlebot/2.1 (+http://www.google.com/bot.html)",
    ]
    for ua in user_agents:
        try:
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if r.status_code == 403:
                continue
            r.raise_for_status()
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(
                ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]
            ):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
            result = "\n".join(lines)[:8000]
            if len(result) > 200:
                return result
        except Exception as exc:
            logger.debug("Scrape attempt failed for %s: %s", url, exc)
    return ""


def _try_download_pdf(url: str, save_path: str, timeout: int = 15) -> bool:
    """Try to download a PDF directly. Returns True if a valid PDF was saved."""
    try:
        assert_url_is_safe(url)
    except SSRFBlockedError as exc:
        logger.info("Refusing PDF URL %s: %s", url, exc)
        return False

    user_agents = [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Googlebot/2.1 (+http://www.google.com/bot.html)",
    ]
    for ua in user_agents:
        try:
            r = requests.get(
                url,
                headers={"User-Agent": ua},
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
            if r.status_code != 200:
                continue
            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                continue
            with open(save_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    fh.write(chunk)
            if Path(save_path).stat().st_size > 1024:
                logger.info("Downloaded PDF %s", save_path)
                return True
            Path(save_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("PDF download failed for %s: %s", url, exc)
    return False


def _is_answer_key_request(query: str) -> bool:
    """Detect if the query is asking for an exam answer key or similar."""
    keywords = (
        "answer key", "answer sheet", "question paper", "question bank",
        "solved paper", "solution pdf", "solutions pdf", "answer pdf",
        "key pdf", "omr sheet", "response sheet", "official key",
        "provisional key", "final key", "shift 1", "shift 2",
        "set a", "set b", "set c", "set d", "paper 1", "paper 2",
    )
    lower = query.lower()
    return any(kw in lower for kw in keywords)


def _is_exam_query(query: str) -> bool:
    """Detect if the query references a known competitive exam."""
    exams = (
        "wbjee", "jee", "neet", "upsc", "ssc", "gate", "cat", "clat",
        "cuet", "bitsat", "viteee", "comedk", "mht-cet", "mhtcet", "kcet",
        "ap eamcet", "ts eamcet", "keam", "cmat", "xat", "snap", "mat",
        "ielts", "toefl", "gre", "gmat", "ugc net", "csir net", "cbse",
        "icse", "isc", "board exam",
    )
    lower = query.lower()
    return any(exam in lower for exam in exams)


def _looks_like_bullets(text: str) -> bool:
    """Return True if the text block looks like a bulleted/numbered list."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return all(re.match(r"^\s*([-*•]|\d+[.)])\s+", line) for line in lines)


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet items from a list-formatted text block."""
    out: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _parse_markdown_blocks(body_text: str, title: str) -> list[Any]:
    """Parse LLM-generated markdown into PDF block elements."""
    blocks: list[Any] = []
    blocks.append(Heading(title, level=1))
    for para in re.split(r"\n{2,}", body_text):
        stripped = para.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            blocks.append(Heading(stripped[3:].strip(), level=2))
        elif stripped.startswith("### "):
            blocks.append(Heading(stripped[4:].strip(), level=3))
        elif _looks_like_bullets(stripped):
            blocks.append(Bullets(_extract_bullets(stripped)))
        else:
            blocks.append(PdfParagraph(stripped))
    return blocks


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def research_and_create_pdf(query: str) -> str:
    """Research a topic across the web and produce a well-formatted PDF.

    Use when the user asks for an answer key, exam result, question paper,
    research summary, or any topic they want compiled into a downloadable PDF.

    Returns a string ending with ``__FILE_PATH__=/tmp/....pdf`` on success,
    or an explanatory message when no useful data was found.
    """
    if not query.strip():
        return "Please tell me what topic you want researched."

    is_answer_key = _is_answer_key_request(query)
    is_exam = _is_exam_query(query)

    # Build search queries
    search_queries = [query]
    if is_answer_key:
        search_queries.extend([
            f"{query} official pdf download",
            f"{query} with solutions",
            f"{query} set wise answers",
        ])
    elif is_exam:
        search_queries.append(f"{query} 2026 official")
    else:
        search_queries.append(f"{query} detailed")

    # Gather search results
    seen_urls: set[str] = set()
    all_results: list[dict[str, Any]] = []
    for q in search_queries[:4]:
        for hit in _ddg_search(q, max_results=8):
            url = hit.get("href") or hit.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(hit)
        if len(all_results) >= 15:
            break

    if not all_results:
        return (
            "I couldn't find any results for this query. "
            "Try rephrasing or being more specific."
        )

    # For answer-key/exam queries, try direct PDF download first
    if is_answer_key or is_exam:
        for hit in all_results[:5]:
            url = hit.get("href") or hit.get("url") or ""
            if url.lower().endswith(".pdf"):
                save_path = _safe_filename(query)
                if _try_download_pdf(url, save_path):
                    return f"Found and downloaded the official PDF. __FILE_PATH__={save_path}"

    # Scrape content from top results
    raw_contents: list[str] = []
    sources_used: list[str] = []
    for hit in all_results[:8]:
        url = hit.get("href") or hit.get("url") or ""
        title = hit.get("title", "Source")
        snippet = hit.get("body", "")
        content = snippet
        if url and not any(url.lower().endswith(x) for x in (".png", ".jpg", ".jpeg", ".gif")):
            scraped = _scrape_url(url, timeout=8)
            if len(scraped) > len(content):
                content = scraped
        if content and len(content) > 50:
            raw_contents.append(f"=== SOURCE: {title} ({url}) ===\n{content[:4000]}")
            sources_used.append(f"{title} — {url}")
        if len(raw_contents) >= 5:
            break

    if not raw_contents:
        return (
            "I searched but couldn't find useful content for this query. "
            "Try being more specific."
        )

    combined = "\n\n".join(raw_contents)

    # Compile with LLM
    if is_answer_key:
        prompt = (
            "You are an expert at extracting answer-key data from web content.\n\n"
            f'The user searched for: "{query}"\n\n'
            "Below is raw content scraped from various websites. Extract the\n"
            "ACTUAL answer-key data if present.\n\n"
            "Format the output as:\n"
            "- Subject-wise sections\n"
            "- Question Number → Answer (A/B/C/D or the actual answer)\n"
            "- Useful adjacent info: expected cutoffs, paper analysis,\n"
            "  important dates, direct download links.\n\n"
            "If the content is ONLY generic articles with no actual data,\n"
            "respond with exactly: NO_USEFUL_DATA\n\n"
            f"Raw content:\n{combined[:12000]}"
        )
        compiled = _llm_compile(prompt)
        if not compiled or "NO_USEFUL_DATA" in compiled:
            links = "\n".join(f"- {s}" for s in sources_used[:5])
            return (
                f"I searched multiple sources but couldn't find the actual "
                f"answer-key data for \"{query}\". The official key may not "
                "be released yet or is only available on the official "
                "website in a format I can't access.\n\n"
                f"Here are some links that might help:\n{links}\n\n"
                "Try checking the official exam-board website directly."
            )
        body_text = compiled
    else:
        prompt = (
            f'Create a well-organized research summary about: "{query}"\n\n'
            "Based on the scraped web content below, produce a comprehensive,\n"
            "well-structured document with clear headings, bullet points and\n"
            "organized sections. Only include factual information found in\n"
            "the sources. Do not make up data.\n\n"
            f"Raw content:\n{combined[:12000]}"
        )
        compiled = _llm_compile(prompt)
        body_text = compiled or "\n\n".join(raw_contents)

    title_for_pdf = query.title()

    # Build PDF
    blocks = _parse_markdown_blocks(body_text, title_for_pdf)
    blocks.append(Rule())
    blocks.append(Heading("Sources", level=2))
    blocks.append(Bullets(sources_used))

    pdf_path = _safe_filename(query)
    try:
        build_pdf(blocks, pdf_path, meta=PdfMeta(title=title_for_pdf, subtitle=""))
    except PdfBuildError as exc:
        return f"Could not build the PDF: {exc}"
    return (
        f"Research compiled from {len(sources_used)} sources into a PDF. "
        f"__FILE_PATH__={pdf_path}"
    )


@tool
def generate_text_to_pdf(text: str, filename: str = "document") -> str:
    """Render free-form text into a clean PDF, honoring Markdown headings.

    Args:
        text: Body content. Lines starting with ``#`` / ``##`` / ``###``
            become headings.
        filename: Used as both the document title and the file stem.

    Returns ``__FILE_PATH__=/tmp/....pdf`` on success.
    """
    if not text.strip():
        return "I need some text to put into the PDF."
    title = filename.replace("_", " ").replace("-", " ").strip().title() or "Document"

    blocks: list[Any] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_body
        if current_heading is not None:
            blocks.append(Heading(current_heading, level=2))
            current_heading = None
        body_text = "\n".join(current_body).strip()
        if body_text:
            for para in re.split(r"\n{2,}", body_text):
                if not para.strip():
                    continue
                if _looks_like_bullets(para):
                    blocks.append(Bullets(_extract_bullets(para)))
                else:
                    blocks.append(PdfParagraph(para.strip()))
        current_body = []

    for line in text.splitlines():
        if line.startswith("### "):
            flush()
            blocks.append(Heading(line[4:].strip(), level=3))
        elif line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
        elif line.startswith("# "):
            flush()
            blocks.append(Heading(line[2:].strip(), level=1))
        else:
            current_body.append(line)
    flush()

    if not blocks:
        blocks.append(PdfParagraph(text))

    pdf_path = _safe_filename(filename)
    try:
        build_pdf(blocks, pdf_path, meta=PdfMeta(title=title, subtitle=""))
    except PdfBuildError as exc:
        return f"Could not build the PDF: {exc}"
    return f"PDF created successfully. __FILE_PATH__={pdf_path}"


# ---------------------------------------------------------------------------
# Legacy aliases — kept for backward compatibility with old prompts /
# saved tool names.
# ---------------------------------------------------------------------------


@tool
def youtube_video_to_pdf(url_or_query: str) -> str:
    """[Deprecated alias] Build a PDF from a YouTube video.

    Delegates to the ``video_to_pdf`` tool in ``full`` mode.
    Prefer ``video_to_pdf`` directly.
    """
    return run_video_to_pdf(url_or_query=url_or_query, mode="full")


@tool
def extract_youtube_to_pdf(url: str) -> str:
    """[Deprecated alias] Extract a YouTube transcript to PDF."""
    return run_video_to_pdf(url_or_query=url, mode="subtitles")


@tool
def search_and_extract_youtube_to_pdf(query: str) -> str:
    """[Deprecated alias] Search YouTube and turn the result into a PDF."""
    return run_video_to_pdf(url_or_query=query, mode="full")
