import os
import re
import time
import urllib.parse
import urllib.request
import logging
import requests
from langchain.tools import tool

logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_filename(name: str, ext=".pdf") -> str:
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:60]
    return f"/tmp/{safe}{ext}"


def _ddg_search(query: str, max_results=8) -> list:
    """DuckDuckGo text search — returns list of {title, url, body}."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.warning(f"DDG search failed: {e}")
        return []


def _scrape_url(url: str, timeout=10) -> str:
    """Scrape visible text from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "lxml")
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)[:8000]
    except Exception as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return ""


def _extract_yt_id(url: str):
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("?")[0]
    if parsed.hostname in ("youtube.com", "www.youtube.com"):
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None


def _get_yt_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(s["text"] for s in segments)
    except Exception as e:
        logger.warning(f"Transcript unavailable for {video_id}: {e}")
        return ""


def _get_yt_metadata(video_id: str) -> dict:
    """Get title/description via yt-dlp (no download)."""
    try:
        import yt_dlp
        opts = {"quiet": True, "skip_download": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return {
                "title": info.get("title", ""),
                "channel": info.get("uploader", ""),
                "description": (info.get("description") or "")[:1000],
                "duration": info.get("duration_string", ""),
                "views": info.get("view_count", 0),
                "upload_date": info.get("upload_date", ""),
            }
    except Exception as e:
        logger.warning(f"yt-dlp metadata failed: {e}")
        return {}


# ─── PDF builder ──────────────────────────────────────────────────────────────

def _build_pdf(title: str, sections: list, pdf_path: str):
    """
    Build a nicely formatted PDF using reportlab.
    sections = list of {"heading": str, "body": str}
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=14,
        spaceAfter=4,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=4,
        textColor=colors.HexColor("#2d2d2d"),
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    story = []

    # Title
    safe_title = title.encode("utf-8", "replace").decode("utf-8")
    story.append(Paragraph(safe_title, title_style))
    story.append(Paragraph(f"Generated by Agent AI Bot • {time.strftime('%d %B %Y')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 12))

    for sec in sections:
        heading = sec.get("heading", "")
        body = sec.get("body", "").strip()
        if not body:
            continue

        if heading:
            story.append(Paragraph(heading.encode("utf-8", "replace").decode("utf-8"), heading_style))

        # Split into paragraphs and render each
        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            # Escape XML special chars for reportlab
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(para.encode("utf-8", "replace").decode("utf-8"), body_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    return pdf_path


# ─── TOOLS ────────────────────────────────────────────────────────────────────

@tool
def research_and_create_pdf(query: str) -> str:
    """
    Researches any topic by searching the internet across multiple sources,
    scrapes and compiles the content, then creates a well-formatted PDF.

    Use this when the user asks for:
    - Answer keys, exam results, question papers (WBJEE, JEE, NEET, etc.)
    - Any topic they want as a PDF/document
    - Research reports, notes, summaries from the web
    - Any factual content gathered from multiple web sources

    This tool searches everywhere — news sites, official boards, educational portals —
    not just one source. It always tries hard to find real content.

    Returns a file path in the format: __FILE_PATH__=/tmp/....pdf
    """
    results_collected = []
    sources_used = []

    # Multiple targeted search queries for better coverage
    queries = [
        query,
        f"{query} 2026",
        f"{query} official answer key",
        f"{query} site result",
    ]

    seen_urls = set()
    all_results = []
    for q in queries[:3]:
        hits = _ddg_search(q, max_results=6)
        for h in hits:
            url = h.get("href", h.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(h)
        if len(all_results) >= 10:
            break

    if not all_results:
        return "Could not find any web results for this query. Try rephrasing."

    # Scrape top results for real content
    for hit in all_results[:6]:
        url = hit.get("href", hit.get("url", ""))
        title = hit.get("title", "Source")
        snippet = hit.get("body", "")

        # Always include the snippet
        content = snippet
        # Try to scrape for more depth (skip heavy PDFs, images)
        if url and not any(url.lower().endswith(x) for x in [".pdf", ".png", ".jpg", ".jpeg", ".gif"]):
            scraped = _scrape_url(url, timeout=8)
            if len(scraped) > len(content):
                content = scraped

        if content:
            results_collected.append({
                "heading": title,
                "body": content[:3000],
            })
            sources_used.append(f"• {title}\n  {url}")

        if len(results_collected) >= 5:
            break

    if not results_collected:
        return "Found links but couldn't extract content. The sites may be blocking scraping."

    # Add sources section at the end
    results_collected.append({
        "heading": "Sources",
        "body": "\n\n".join(sources_used),
    })

    pdf_path = _safe_filename(query)
    _build_pdf(
        title=query,
        sections=results_collected,
        pdf_path=pdf_path,
    )

    return f"Research complete. Compiled content from {len(sources_used)} sources into a PDF. __FILE_PATH__={pdf_path}"


@tool
def youtube_video_to_pdf(url_or_query: str) -> str:
    """
    Takes a YouTube URL OR a search query, finds the video, extracts the full
    transcript, gets metadata, and creates a well-formatted PDF summary.

    Use when user wants to:
    - Summarize a YouTube video (with or without URL)
    - Get a YouTube video transcript as PDF
    - "Turn this YouTube video into PDF"
    - "Summarize this YT video: [url]"

    Returns a file path in the format: __FILE_PATH__=/tmp/....pdf
    """
    video_id = None
    video_url = url_or_query
    meta = {}

    # Check if it's a URL
    if "youtube.com" in url_or_query or "youtu.be" in url_or_query:
        video_id = _extract_yt_id(url_or_query)
    else:
        # Search for the video
        hits = _ddg_search(f"site:youtube.com {url_or_query}", max_results=5)
        for hit in hits:
            url = hit.get("href", "")
            vid = _extract_yt_id(url)
            if vid:
                video_id = vid
                video_url = url
                break

    if not video_id:
        return f"Could not find a YouTube video for: {url_or_query}"

    # Get metadata and transcript in parallel-ish
    meta = _get_yt_metadata(video_id)
    transcript = _get_yt_transcript(video_id)

    title = meta.get("title") or f"YouTube Video ({video_id})"

    sections = []

    # Video info section
    info_lines = []
    if meta.get("channel"):
        info_lines.append(f"Channel: {meta['channel']}")
    if meta.get("duration"):
        info_lines.append(f"Duration: {meta['duration']}")
    if meta.get("views"):
        info_lines.append(f"Views: {meta['views']:,}")
    if meta.get("upload_date"):
        d = meta["upload_date"]
        info_lines.append(f"Upload date: {d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d)
    info_lines.append(f"URL: {video_url}")

    if info_lines:
        sections.append({"heading": "Video Info", "body": "\n".join(info_lines)})

    if meta.get("description"):
        sections.append({"heading": "Description", "body": meta["description"]})

    if transcript:
        # Break transcript into readable chunks (~800 chars each)
        chunks = []
        words = transcript.split()
        chunk = []
        char_count = 0
        for word in words:
            chunk.append(word)
            char_count += len(word) + 1
            if char_count >= 800:
                chunks.append(" ".join(chunk))
                chunk = []
                char_count = 0
        if chunk:
            chunks.append(" ".join(chunk))

        sections.append({
            "heading": "Full Transcript",
            "body": "\n\n".join(chunks),
        })
    else:
        sections.append({
            "heading": "Note",
            "body": "This video does not have an available transcript. Only metadata could be extracted.",
        })

    pdf_path = _safe_filename(f"yt_{video_id}")
    _build_pdf(title=title, sections=sections, pdf_path=pdf_path)

    return f"YouTube video '{title}' processed. PDF generated with transcript and metadata. __FILE_PATH__={pdf_path}"


@tool
def generate_text_to_pdf(text: str, filename: str = "document") -> str:
    """
    Creates a well-formatted PDF from provided text content.
    Use when you have the content ready and need to write it to a PDF file.
    Supports markdown-style headings (lines starting with # or ##).
    Returns: __FILE_PATH__=/tmp/....pdf
    """
    sections = []
    current_heading = ""
    current_body = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_body:
                sections.append({"heading": current_heading, "body": "\n".join(current_body)})
                current_body = []
            current_heading = line[3:].strip()
        elif line.startswith("# "):
            if current_body:
                sections.append({"heading": current_heading, "body": "\n".join(current_body)})
                current_body = []
            current_heading = line[2:].strip()
        else:
            current_body.append(line)

    if current_body:
        sections.append({"heading": current_heading, "body": "\n".join(current_body)})

    if not sections:
        sections = [{"heading": "", "body": text}]

    pdf_path = _safe_filename(filename)
    _build_pdf(title=filename.replace("_", " ").replace("-", " ").title(), sections=sections, pdf_path=pdf_path)

    return f"PDF created successfully. __FILE_PATH__={pdf_path}"


# Keep old tools as aliases for backward compat
@tool
def extract_youtube_to_pdf(url: str) -> str:
    """
    Extract a YouTube video transcript and save to PDF. Alias for youtube_video_to_pdf.
    Use when user provides a YouTube URL directly.
    Returns: __FILE_PATH__=/tmp/....pdf
    """
    return youtube_video_to_pdf.invoke(url)


@tool
def search_and_extract_youtube_to_pdf(query: str) -> str:
    """
    Search for a YouTube video and extract its transcript to PDF.
    Use when user wants a YouTube summary but gives a topic not a URL.
    Returns: __FILE_PATH__=/tmp/....pdf
    """
    return youtube_video_to_pdf.invoke(query)
