import os
import re
import time
import urllib.parse
import urllib.request
import logging
import requests
import json
from langchain.tools import tool

logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_filename(name: str, ext=".pdf") -> str:
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:60]
    return f"/tmp/{safe}{ext}"


def _ddg_search(query: str, max_results=8) -> list:
    """DuckDuckGo/ddgs text search — returns list of {title, url, body}."""
    for pkg in ["ddgs", "duckduckgo_search"]:
        try:
            if pkg == "ddgs":
                from ddgs import DDGS
            else:
                from duckduckgo_search import DDGS
            with DDGS() as d:
                results = list(d.text(query, max_results=max_results))
            if results:
                return results
        except Exception as e:
            logger.debug(f"Search via {pkg} failed: {e}")
    # Last resort: Bing scrape
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count={max_results}"
        r = requests.get(url, headers=headers, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for li in soup.select("li.b_algo")[:max_results]:
            a = li.select_one("h2 a")
            snippet = li.select_one(".b_caption p")
            if a:
                results.append({
                    "title": a.get_text(strip=True),
                    "href": a.get("href", ""),
                    "body": snippet.get_text(strip=True) if snippet else "",
                })
        return results
    except Exception as e:
        logger.warning(f"Bing fallback search failed: {e}")
    return []


def _scrape_url(url: str, timeout=10) -> str:
    """Scrape visible text from a URL with multiple user-agent fallbacks."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
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
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
            result = "\n".join(lines)[:8000]
            if len(result) > 200:
                return result
        except Exception as e:
            logger.debug(f"Scrape attempt failed for {url} with UA {ua[:30]}: {e}")
    logger.warning(f"All scrape attempts failed for {url}")
    return ""


def _try_download_pdf(url: str, save_path: str, timeout=15) -> bool:
    """Try to download a PDF file directly from a URL."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Googlebot/2.1 (+http://www.google.com/bot.html)",
    ]
    for ua in user_agents:
        try:
            headers = {"User-Agent": ua}
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "")
                if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    size = os.path.getsize(save_path)
                    if size > 1000:  # At least 1KB
                        logger.info(f"Downloaded PDF: {save_path} ({size} bytes)")
                        return True
                    else:
                        os.remove(save_path)
        except Exception as e:
            logger.debug(f"PDF download failed for {url}: {e}")
    return False


def _is_answer_key_request(query: str) -> bool:
    """Detect if the user is asking for an answer key / question paper / specific exam document."""
    keywords = [
        "answer key", "answer sheet", "question paper", "question bank",
        "solved paper", "solution pdf", "solutions pdf", "answer pdf",
        "key pdf", "omr sheet", "response sheet", "official key",
        "provisional key", "final key", "shift 1", "shift 2",
        "set a", "set b", "set c", "set d", "paper 1", "paper 2",
    ]
    lower = query.lower()
    return any(kw in lower for kw in keywords)


def _is_exam_query(query: str) -> bool:
    """Check if query is about an exam."""
    exams = [
        "wbjee", "jee", "neet", "upsc", "ssc", "gate", "cat", "clat",
        "cuet", "bitsat", "viteee", "comedk", "mht-cet", "mhtcet",
        "kcet", "ap eamcet", "ts eamcet", "keam", "cmat", "xat",
        "snap", "mat", "ielts", "toefl", "gre", "gmat", "ugc net",
        "csir net", "cbse", "icse", "isc", "board exam",
    ]
    lower = query.lower()
    return any(exam in lower for exam in exams)


def _gemini_compile_answer_key(raw_content: str, query: str) -> str:
    """Use Gemini to extract and structure answer key data from raw scraped content."""
    try:
        from config import load_config
        config = load_config()
        api_key = config.get("gemini_api_key", "")
        if not api_key:
            return ""

        prompt = f"""You are an expert at extracting answer key data from web content.

The user searched for: "{query}"

Below is raw content scraped from various websites. Extract the ACTUAL answer key data if present.

Format the output as:
- Subject-wise sections
- Question Number → Answer (A/B/C/D or the actual answer)
- If exact answers aren't in the content, extract any useful information like:
  - Expected cutoff marks
  - Paper analysis (difficulty level, topic-wise breakdown)
  - Important dates (when official key releases)
  - Direct download links for answer key PDFs

If the content is ONLY generic articles about the exam with NO actual answer data, question-answer mappings, cutoff info, or paper analysis, respond with exactly: NO_USEFUL_DATA

Raw content:
{raw_content[:12000]}"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4000}
        }
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if text and "NO_USEFUL_DATA" not in text:
                return text
            logger.info("Gemini determined: no useful answer key data in scraped content")
        else:
            logger.warning(f"Gemini API error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.warning(f"Gemini compile failed: {e}")
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

    safe_title = title.encode("utf-8", "replace").decode("utf-8")
    story.append(Paragraph(safe_title, title_style))
    story.append(Paragraph(f"Generated by Agent AI Bot &bull; {time.strftime('%d %B %Y')}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 12))

    for sec in sections:
        heading = sec.get("heading", "")
        body = sec.get("body", "").strip()
        if not body:
            continue

        if heading:
            safe_h = heading.encode("utf-8", "replace").decode("utf-8")
            safe_h = safe_h.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_h, heading_style))

        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
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

    This tool searches everywhere — news sites, official boards, educational portals.
    For answer key requests, it tries to find and download actual answer key PDFs,
    extract real Q→Answer data, and uses AI to compile structured results.

    Returns a file path in the format: __FILE_PATH__=/tmp/....pdf
    OR a text message if no useful data was found.
    """
    is_answer_key = _is_answer_key_request(query)
    is_exam = _is_exam_query(query)

    # ── Step 1: Search with multiple targeted queries ──
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

    seen_urls = set()
    all_results = []
    for q in search_queries[:4]:
        hits = _ddg_search(q, max_results=8)
        for h in hits:
            url = h.get("href", h.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(h)
        if len(all_results) >= 15:
            break

    if not all_results:
        return "I couldn't find any results for this query. Try rephrasing or being more specific."

    # ── Step 2: Try to find and download actual PDF files ──
    if is_answer_key or is_exam:
        pdf_urls = []
        for hit in all_results:
            url = hit.get("href", hit.get("url", ""))
            title = hit.get("title", "").lower()
            body = hit.get("body", "").lower()
            # Look for direct PDF links
            if url.lower().endswith(".pdf"):
                pdf_urls.append(url)
            # Look for links that suggest PDF downloads
            if any(kw in title + body for kw in ["download pdf", "answer key pdf", "download answer key", "official answer key"]):
                pdf_urls.append(url)

        # Try downloading actual PDFs
        for pdf_url in pdf_urls[:3]:
            if pdf_url.lower().endswith(".pdf"):
                save_path = _safe_filename(query)
                if _try_download_pdf(pdf_url, save_path):
                    return f"Found and downloaded the official PDF. __FILE_PATH__={save_path}"

    # ── Step 3: Scrape content from top results ──
    raw_contents = []
    sources_used = []

    for hit in all_results[:8]:
        url = hit.get("href", hit.get("url", ""))
        title = hit.get("title", "Source")
        snippet = hit.get("body", "")

        content = snippet
        if url and not any(url.lower().endswith(x) for x in [".pdf", ".png", ".jpg", ".jpeg", ".gif"]):
            scraped = _scrape_url(url, timeout=8)
            if len(scraped) > len(content):
                content = scraped

        if content and len(content) > 50:
            raw_contents.append(f"=== SOURCE: {title} ({url}) ===\n{content[:4000]}")
            sources_used.append(f"• {title}\n  {url}")

        if len(raw_contents) >= 5:
            break

    # ── Step 4: For answer key requests, use Gemini to extract structured data ──
    if is_answer_key and raw_contents:
        combined_raw = "\n\n".join(raw_contents)
        compiled = _gemini_compile_answer_key(combined_raw, query)

        if compiled:
            # Gemini found useful data — build PDF
            sections = [
                {"heading": query.title(), "body": compiled},
                {"heading": "Sources", "body": "\n\n".join(sources_used)},
            ]
            pdf_path = _safe_filename(query)
            _build_pdf(title=query.title(), sections=sections, pdf_path=pdf_path)
            return f"Answer key compiled from {len(sources_used)} sources. __FILE_PATH__={pdf_path}"
        else:
            # Gemini says no useful answer key data found
            logger.info(f"No actual answer key data found for: {query}")
            # Collect useful links to mention
            links_text = "\n".join(f"• {h.get('title','')}: {h.get('href', h.get('url',''))}" for h in all_results[:5])
            return (
                f"I searched multiple sources but couldn't find the actual answer key data for \"{query}\". "
                f"The official answer key may not have been released yet, or it's only available on the official website in a format I can't access.\n\n"
                f"Here are some links that might help:\n{links_text}\n\n"
                f"Try checking the official exam board website directly."
            )

    # ── Step 5: For general research, build PDF from scraped content ──
    if not raw_contents:
        # Use snippets as fallback for non-answer-key requests
        for hit in all_results[:8]:
            title = hit.get("title", "Source")
            body = hit.get("body", hit.get("snippet", ""))
            url = hit.get("href", hit.get("url", ""))
            if body:
                raw_contents.append(f"=== SOURCE: {title} ({url}) ===\n{body}")
                sources_used.append(f"• {title}\n  {url}")

    if not raw_contents:
        return "I searched but couldn't find useful content for this query. Try being more specific."

    # For general research PDFs — use Gemini to create a well-structured summary
    combined_raw = "\n\n".join(raw_contents)
    try:
        from config import load_config
        config = load_config()
        api_key = config.get("gemini_api_key", "")
        if api_key:
            prompt = f"""Create a well-organized research summary about: "{query}"

Based on the following scraped web content, create a comprehensive and well-structured document.
Use clear headings, bullet points, and organized sections.
Only include factual information found in the sources. Do not make up data.

Raw content:
{combined_raw[:12000]}"""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4000}
            }
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                summary = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if summary and len(summary) > 100:
                    sections = [
                        {"heading": "", "body": summary},
                        {"heading": "Sources", "body": "\n\n".join(sources_used)},
                    ]
                    pdf_path = _safe_filename(query)
                    _build_pdf(title=query.title(), sections=sections, pdf_path=pdf_path)
                    return f"Research compiled from {len(sources_used)} sources into a PDF. __FILE_PATH__={pdf_path}"
    except Exception as e:
        logger.warning(f"Gemini summary failed: {e}")

    # Fallback: raw content PDF
    sections = []
    for raw in raw_contents:
        lines = raw.split("\n", 1)
        heading = lines[0].replace("=== SOURCE: ", "").replace(" ===", "") if lines else "Source"
        body = lines[1] if len(lines) > 1 else raw
        sections.append({"heading": heading, "body": body[:3000]})
    sections.append({"heading": "Sources", "body": "\n\n".join(sources_used)})

    pdf_path = _safe_filename(query)
    _build_pdf(title=query.title(), sections=sections, pdf_path=pdf_path)
    return f"Research compiled from {len(sources_used)} sources into a PDF. __FILE_PATH__={pdf_path}"


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

    if "youtube.com" in url_or_query or "youtu.be" in url_or_query:
        video_id = _extract_yt_id(url_or_query)
    else:
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

    meta = _get_yt_metadata(video_id)
    transcript = _get_yt_transcript(video_id)

    title = meta.get("title") or f"YouTube Video ({video_id})"

    sections = []

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
