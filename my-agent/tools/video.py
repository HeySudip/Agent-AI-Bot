"""Video-to-PDF tool.

Modes: subtitles, screenshots, summary, qa, full.

Pipeline:
1. Resolve URL / search query → video_id, canonical URL, metadata.
2. Fetch transcript (youtube-transcript-api v1.2.4).
3. (Optional) Download video via yt-dlp, extract frames via ffmpeg.
4. (Optional) LLM summarization/QA via llm_provider.generate_text.
5. Render PDF via pdf_builder.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests
from langchain.tools import tool

from safety.ssrf_guard import SSRFBlockedError, assert_url_is_safe

from .pdf_builder import (
    Bullets,
    Heading,
    KeyValue,
    PageBreak,
    PdfBuildError,
    PdfMeta,
    Rule,
    build_pdf,
)
from .pdf_builder import Image as PdfImage
from .pdf_builder import Paragraph as PdfParagraph

logger = logging.getLogger(__name__)

VideoMode = Literal["subtitles", "screenshots", "summary", "qa", "full"]
_VALID_MODES: tuple[str, ...] = ("subtitles", "screenshots", "summary", "qa", "full")

_DEFAULT_FRAMES = 8
_MAX_FRAMES = 24
_MIN_FRAMES = 1
_VIDEO_DOWNLOAD_TIMEOUT_S = 180
_FRAME_EXTRACTION_TIMEOUT_S = 90

_TRANSCRIPT_PREFERRED_LANGS = [
    "en", "en-US", "en-GB", "hi", "es", "fr", "de", "pt", "ru", "ja", "ko", "zh", "ar", "id",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video."""

    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    duration_s: int = 0
    duration_str: str = ""
    views: int = 0
    upload_date: str = ""
    description: str = ""
    thumbnail_url: str = ""


@dataclass
class TranscriptResult:
    """Result of a transcript fetch attempt."""

    text: str = ""
    language: str = ""
    is_generated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


@dataclass
class FramesResult:
    """Result of frame extraction."""

    frame_paths: list[str] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.frame_paths)


@dataclass
class LLMResult:
    """Result of an LLM call."""

    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")

    if host in {"youtu.be", "youtu.be."}:
        return parsed.path.lstrip("/").split("/")[0] or None
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
        for prefix in ("/shorts/", "/embed/"):
            if parsed.path.startswith(prefix):
                parts = parsed.path.split("/")
                return parts[2] if len(parts) > 2 and parts[2] else None
    return None


def canonical_url(video_id: str) -> str:
    """Return the canonical YouTube watch URL."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration string."""
    if not seconds or seconds < 0:
        return ""
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


# ---------------------------------------------------------------------------
# Metadata + search
# ---------------------------------------------------------------------------


def _resolve_query_to_url(query: str) -> str | None:
    """Search DuckDuckGo for a YouTube video matching the query."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    try:
        with DDGS() as d:
            candidates = list(d.text(f"site:youtube.com {query}", max_results=8))
    except Exception as exc:
        logger.info("DDG search failed: %s", exc)
        return None

    for hit in candidates:
        href = hit.get("href") or hit.get("url") or ""
        if extract_video_id(href):
            return href
    return None


def fetch_metadata(video_id: str, url: str) -> VideoMetadata:
    """Fetch video metadata via yt-dlp. Gracefully handles missing yt-dlp."""
    meta = VideoMetadata(
        video_id=video_id,
        url=url,
        thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    )
    try:
        import yt_dlp
    except ImportError:
        logger.info("yt-dlp not installed; skipping metadata fetch")
        return meta

    opts: dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,
        "extractor_args": {"youtube": {"player_client": ["tv_embedded"]}},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        logger.info("yt-dlp metadata extract failed: %s", exc)
        return meta

    meta.title = (info.get("title") or "").strip()
    meta.channel = (info.get("uploader") or info.get("channel") or "").strip()
    meta.duration_s = int(info.get("duration") or 0)
    meta.duration_str = info.get("duration_string") or _format_duration(meta.duration_s)
    meta.views = int(info.get("view_count") or 0)
    meta.upload_date = info.get("upload_date") or ""
    meta.description = (info.get("description") or "").strip()[:1500]
    return meta


# ---------------------------------------------------------------------------
# Transcript fetching (youtube-transcript-api v1.2.4)
# ---------------------------------------------------------------------------


def fetch_transcript(video_id: str, preferred_languages: list[str] | None = None) -> TranscriptResult:
    """Fetch transcript using youtube-transcript-api v1.2.4 API.

    Tries manual transcripts in preferred languages first, then auto-generated,
    then any available language (translated to English when possible).
    """
    languages = preferred_languages or _TRANSCRIPT_PREFERRED_LANGS

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        return TranscriptResult(error=f"youtube_transcript_api not installed: {exc}")

    ytt = YouTubeTranscriptApi()

    # List available transcripts
    try:
        transcript_list = ytt.list(video_id)
    except Exception as exc:
        err_str = str(exc).lower()
        if "disabled" in err_str or "no transcript" in err_str:
            return TranscriptResult(error="No transcripts are available for this video.")
        if "unavailable" in err_str or "private" in err_str:
            return TranscriptResult(error="The video is unavailable or private.")
        return TranscriptResult(error=f"Failed to list transcripts: {exc}")

    selected = None
    selected_lang = ""
    is_generated = False

    # 1) Manual transcript in a preferred language
    try:
        selected = transcript_list.find_manually_created_transcript(languages)
        selected_lang = selected.language_code
    except Exception:
        selected = None

    # 2) Auto-generated in a preferred language
    if selected is None:
        try:
            selected = transcript_list.find_generated_transcript(languages)
            selected_lang = selected.language_code
            is_generated = True
        except Exception:
            selected = None

    # 3) Any available transcript, translate to English if possible
    if selected is None:
        try:
            for tr in transcript_list:
                selected = tr
                selected_lang = tr.language_code
                is_generated = bool(getattr(tr, "is_generated", False))
                break
        except Exception:
            pass

    if selected is None:
        return TranscriptResult(error="No transcripts could be selected for this video.")

    # Fetch the transcript content
    target_lang = selected_lang
    try:
        if selected_lang not in languages:
            try:
                translated = selected.translate("en")
                fetched = translated.fetch()
                target_lang = "en (translated)"
            except Exception:
                fetched = selected.fetch()
        else:
            fetched = selected.fetch()
    except Exception as exc:
        return TranscriptResult(error=f"Transcript fetch failed: {exc}")

    # Extract text from snippets
    text_parts: list[str] = []
    for snippet in fetched:
        t = getattr(snippet, "text", "") or (snippet.get("text", "") if isinstance(snippet, dict) else "")
        if t.strip():
            text_parts.append(t.strip())

    text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
    return TranscriptResult(text=text, language=target_lang, is_generated=is_generated)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _ffmpeg_binary() -> str | None:
    """Return path to ffmpeg, preferring imageio-ffmpeg's vendored binary."""
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and Path(path).exists():
            return path
    except ImportError:
        pass
    return shutil.which("ffmpeg")


def _download_video(url: str, work_dir: Path) -> tuple[Path | None, str]:
    """Download lowest-quality video via yt-dlp. Returns (path, error)."""
    try:
        import yt_dlp
    except ImportError:
        return None, "yt-dlp is not installed; cannot download video for screenshots."

    out_template = str(work_dir / "video.%(ext)s")
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "format": "worst[ext=mp4]/worst",
        "outtmpl": out_template,
        "noplaylist": True,
        "concurrent_fragment_downloads": 1,
        "socket_timeout": 30,
        "retries": 2,
        "extractor_args": {"youtube": {"player_client": ["tv_embedded"]}},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        return None, f"yt-dlp download failed: {exc}"

    candidates = sorted(work_dir.glob("video.*"))
    if candidates:
        return candidates[0], ""
    return None, "Video file not produced by yt-dlp."


def _download_thumbnail(url: str, dest: Path, timeout: float = 15.0) -> bool:
    """Download a thumbnail image, respecting SSRF guard."""
    try:
        assert_url_is_safe(url)
    except SSRFBlockedError as exc:
        logger.info("Refusing thumbnail URL: %s", exc)
        return False
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False
        dest.write_bytes(resp.content)
        return dest.stat().st_size > 1024
    except requests.RequestException as exc:
        logger.info("Thumbnail download failed: %s", exc)
        return False


def extract_frames(
    video_url: str,
    *,
    n_frames: int,
    work_dir: Path,
    duration_s: int = 0,
    thumbnail_url: str = "",
) -> FramesResult:
    """Extract evenly-spaced frames from a video.

    Falls back to YouTube thumbnail if yt-dlp/ffmpeg unavailable.
    """
    n_frames = max(_MIN_FRAMES, min(_MAX_FRAMES, n_frames))
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path, dl_error = _download_video(video_url, work_dir)
    ffmpeg = _ffmpeg_binary()

    if not video_path or not ffmpeg:
        if thumbnail_url:
            thumb = work_dir / "thumb.jpg"
            if _download_thumbnail(thumbnail_url, thumb):
                return FramesResult(
                    frame_paths=[str(thumb)],
                    timestamps_s=[0.0],
                    error=dl_error or "ffmpeg unavailable; using thumbnail only.",
                )
        return FramesResult(error=dl_error or "Cannot extract frames: yt-dlp or ffmpeg missing.")

    # Compute capture timestamps
    if duration_s and duration_s > 0:
        total = max(1, duration_s - 1)
        if n_frames == 1:
            timestamps = [total / 2.0]
        else:
            step = total / (n_frames + 1)
            timestamps = [step * (i + 1) for i in range(n_frames)]
    else:
        timestamps = [(i + 1) * 5.0 for i in range(n_frames)]

    frame_paths: list[str] = []
    for i, ts in enumerate(timestamps, start=1):
        out_path = work_dir / f"frame_{i:03d}.jpg"
        cmd = [
            ffmpeg, "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "3", "-loglevel", "error", str(out_path),
        ]
        try:
            subprocess.run(
                cmd, check=False, timeout=_FRAME_EXTRACTION_TIMEOUT_S,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            logger.info("ffmpeg frame extraction timed out at %.2fs", ts)
            continue
        if out_path.exists() and out_path.stat().st_size > 1024:
            frame_paths.append(str(out_path))

    if not frame_paths:
        if thumbnail_url:
            thumb = work_dir / "thumb.jpg"
            if _download_thumbnail(thumbnail_url, thumb):
                return FramesResult(
                    frame_paths=[str(thumb)],
                    timestamps_s=[0.0],
                    error="ffmpeg produced no frames; using thumbnail.",
                )
        return FramesResult(error="ffmpeg produced no frames from the downloaded video.")

    return FramesResult(frame_paths=frame_paths, timestamps_s=timestamps[:len(frame_paths)])


# ---------------------------------------------------------------------------
# LLM summarization / Q&A via llm_provider
# ---------------------------------------------------------------------------


def _llm_generate(prompt: str, *, max_tokens: int = 2048, temperature: float = 0.2) -> LLMResult:
    """Call llm_provider.generate_text synchronously. Handles key rotation + Groq fallback."""
    try:
        from llm_provider import generate_text
    except ImportError as exc:
        return LLMResult(error=f"llm_provider module not available: {exc}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            import concurrent.futures
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
        return LLMResult(error=f"LLM generation failed: {exc}")

    return LLMResult(text=text) if text.strip() else LLMResult(error="LLM returned empty text.")


def summarize_transcript(meta: VideoMetadata, transcript: str) -> LLMResult:
    """Generate a structured summary of the video transcript."""
    if not transcript.strip():
        return LLMResult(error="Empty transcript; nothing to summarize.")
    prompt = (
        "You are creating a concise, well-structured summary of a video.\n\n"
        f"Title: {meta.title or '(unknown)'}\n"
        f"Channel: {meta.channel or '(unknown)'}\n"
        f"Duration: {meta.duration_str or '(unknown)'}\n\n"
        "Produce the summary using exactly this Markdown layout:\n\n"
        "## Overview\nTwo or three sentences describing what the video is about.\n\n"
        "## Key points\nFive to ten bullet points capturing the main ideas, in order.\n\n"
        "## Notable quotes or facts\nUp to five short quoted lines worth highlighting.\n\n"
        "## Takeaways\nThree concrete, actionable takeaways for the viewer.\n\n"
        "Only use facts present in the transcript. Do not invent information.\n\n"
        f"TRANSCRIPT:\n{transcript[:18000]}"
    )
    return _llm_generate(prompt, max_tokens=2048, temperature=0.2)


def answer_question(meta: VideoMetadata, transcript: str, question: str) -> LLMResult:
    """Answer a question about the video using only its transcript."""
    if not transcript.strip():
        return LLMResult(error="Empty transcript; cannot answer.")
    if not question.strip():
        return LLMResult(error="Empty question.")
    prompt = (
        "You are answering questions about a video using ONLY its transcript.\n\n"
        f"Video title: {meta.title or '(unknown)'}\n\n"
        f"Question: {question.strip()}\n\n"
        "Instructions:\n"
        "- Answer in 1-3 short paragraphs.\n"
        "- If the transcript does not contain enough information, say so plainly.\n"
        "- Quote the transcript briefly when it directly supports your answer.\n"
        "- Do not invent facts.\n\n"
        f"TRANSCRIPT:\n{transcript[:18000]}"
    )
    return _llm_generate(prompt, max_tokens=1024, temperature=0.1)


# ---------------------------------------------------------------------------
# PDF block assembly
# ---------------------------------------------------------------------------


def _parse_question_list(raw: str) -> list[str]:
    """Parse newline/semicolon-separated questions."""
    if not raw:
        return []
    parts = [p.strip(" -•\t").strip() for p in re.split(r"[\n;]+|(?<=\?)\s+", raw)]
    return [p for p in parts if len(p) > 2][:5]


def _safe_filename(seed: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", seed).strip("_")[:60] or "video"
    return f"{base}_{int(time.time())}.pdf"


def _output_dir() -> Path:
    out = Path(tempfile.gettempdir()) / "agent_pdfs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _transcript_blocks(transcript: str, *, paragraph_chars: int = 700) -> list[Any]:
    """Chunk transcript text into readable paragraphs for PDF rendering."""
    if not transcript:
        return []
    words = transcript.split()
    paragraphs: list[str] = []
    buf: list[str] = []
    count = 0
    for w in words:
        buf.append(w)
        count += len(w) + 1
        if count >= paragraph_chars:
            paragraphs.append(" ".join(buf))
            buf = []
            count = 0
    if buf:
        paragraphs.append(" ".join(buf))
    return [PdfParagraph(p) for p in paragraphs]


def _meta_rows(meta: VideoMetadata) -> list[tuple[str, str]]:
    """Build key-value rows for the metadata table."""
    rows: list[tuple[str, str]] = []
    if meta.title:
        rows.append(("Title", meta.title))
    if meta.channel:
        rows.append(("Channel", meta.channel))
    if meta.duration_str:
        rows.append(("Duration", meta.duration_str))
    if meta.upload_date and len(meta.upload_date) == 8:
        d = meta.upload_date
        rows.append(("Upload date", f"{d[:4]}-{d[4:6]}-{d[6:]}"))
    elif meta.upload_date:
        rows.append(("Upload date", meta.upload_date))
    if meta.views:
        rows.append(("Views", f"{meta.views:,}"))
    rows.append(("URL", meta.url))
    return rows


def _looks_like_bullets(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return all(re.match(r"^\s*([-*•]|\d+[.)])\s+", line) for line in lines)


def _extract_bullets(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", line).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _processing_notes(
    *,
    transcript: TranscriptResult,
    summary: LLMResult,
    frames: FramesResult,
    qa: list[tuple[str, LLMResult]],
    mode: VideoMode,
) -> list[str]:
    """Collect processing notes for the PDF footer."""
    notes: list[str] = []
    if mode in ("subtitles", "summary", "qa", "full"):
        if transcript.ok:
            notes.append(f"Transcript: ok ({transcript.language})")
        elif transcript.error:
            notes.append(f"Transcript: skipped — {transcript.error}")
    if mode in ("summary", "full") and summary.error:
        notes.append(f"Summary: skipped — {summary.error}")
    if mode in ("screenshots", "full") and frames.error:
        notes.append(f"Screenshots: degraded — {frames.error}")
    for q, res in qa:
        if not res.ok and res.error:
            notes.append(f"Q ('{q[:40]}…'): skipped — {res.error}")
    return notes


def _build_blocks(
    *,
    meta: VideoMetadata,
    mode: VideoMode,
    transcript: TranscriptResult,
    summary: LLMResult,
    qa: list[tuple[str, LLMResult]],
    frames: FramesResult,
) -> list[Any]:
    """Assemble the ordered list of PDF blocks for the given mode."""
    blocks: list[Any] = []

    blocks.append(KeyValue(rows=_meta_rows(meta)))
    if meta.description:
        blocks.append(Heading("Description", level=2))
        blocks.append(PdfParagraph(meta.description))
    blocks.append(Rule())

    # Summary section
    if mode in ("summary", "full"):
        blocks.append(Heading("Summary", level=1))
        if summary.ok:
            for para in re.split(r"\n{2,}", summary.text):
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
        else:
            blocks.append(PdfParagraph(summary.error or "Summary unavailable for this video."))

    # Q&A section
    if mode in ("qa", "full") and qa:
        blocks.append(PageBreak())
        blocks.append(Heading("Questions & Answers", level=1))
        for question, result in qa:
            blocks.append(Heading(question, level=2))
            blocks.append(PdfParagraph(result.text if result.ok else (result.error or "No answer.")))

    # Screenshots section
    if mode in ("screenshots", "full"):
        blocks.append(PageBreak())
        blocks.append(Heading("Screenshots", level=1))
        if frames.ok:
            for i, frame in enumerate(frames.frame_paths, start=1):
                ts = frames.timestamps_s[i - 1] if i - 1 < len(frames.timestamps_s) else 0.0
                caption = f"Frame {i} · t = {_format_duration(int(ts))}" if ts else f"Frame {i}"
                blocks.append(PdfImage(path=frame, caption=caption))
        else:
            blocks.append(PdfParagraph(frames.error or "No screenshots could be extracted."))

    # Transcript section
    if mode in ("subtitles", "full"):
        blocks.append(PageBreak())
        lang = transcript.language or "unknown"
        kind = "auto-generated" if transcript.is_generated else "manual"
        blocks.append(Heading(f"Transcript ({lang}, {kind})", level=1))
        if transcript.ok:
            blocks.extend(_transcript_blocks(transcript.text))
        else:
            blocks.append(PdfParagraph(transcript.error or "Transcript unavailable for this video."))

    # Processing notes
    notes = _processing_notes(transcript=transcript, summary=summary, frames=frames, qa=qa, mode=mode)
    if notes:
        blocks.append(Rule())
        blocks.append(Heading("Processing notes", level=3))
        blocks.append(Bullets(notes))

    return blocks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_video_to_pdf(
    url_or_query: str,
    *,
    mode: VideoMode = "full",
    questions: str = "",
    n_frames: int = _DEFAULT_FRAMES,
) -> str:
    """Main driver. Returns a message with __FILE_PATH__ on success or an error string."""
    if mode not in _VALID_MODES:
        return f"Invalid mode '{mode}'. Use one of: {', '.join(_VALID_MODES)}."

    target = url_or_query.strip()
    if not target:
        return "Please provide a YouTube URL or a search query."

    video_id = extract_video_id(target)
    if not video_id:
        resolved = _resolve_query_to_url(target)
        if resolved:
            video_id = extract_video_id(resolved)
            target = resolved
    if not video_id:
        return f"Could not find a YouTube video for: {url_or_query}"

    url = canonical_url(video_id)
    meta = fetch_metadata(video_id, url)

    # Transcript
    transcript_result = TranscriptResult()
    if mode in ("subtitles", "summary", "qa", "full"):
        transcript_result = fetch_transcript(video_id)

    # Q&A
    questions_list: list[str] = []
    qa_results: list[tuple[str, LLMResult]] = []
    if mode in ("qa", "full"):
        questions_list = _parse_question_list(questions) if questions else []
        if mode == "qa" and not questions_list:
            return "Please provide one or more questions for QA mode."
        for q in questions_list:
            qa_results.append((q, answer_question(meta, transcript_result.text, q)))

    # Summary
    summary_result = LLMResult()
    if mode in ("summary", "full") and transcript_result.ok:
        summary_result = summarize_transcript(meta, transcript_result.text)

    # Frames
    frames_result = FramesResult()
    if mode in ("screenshots", "full"):
        with tempfile.TemporaryDirectory(prefix="vid_frames_") as td:
            work = Path(td)
            frames_result = extract_frames(
                url,
                n_frames=n_frames,
                work_dir=work,
                duration_s=meta.duration_s,
                thumbnail_url=meta.thumbnail_url,
            )
            # Persist frames outside the temp dir
            persistent_dir = _output_dir() / f"frames_{video_id}_{int(time.time())}"
            persistent_dir.mkdir(parents=True, exist_ok=True)
            persisted: list[str] = []
            for fp in frames_result.frame_paths:
                src = Path(fp)
                if src.exists():
                    dest = persistent_dir / src.name
                    shutil.copyfile(src, dest)
                    persisted.append(str(dest))
            frames_result.frame_paths = persisted

    # Build PDF
    blocks = _build_blocks(
        meta=meta, mode=mode, transcript=transcript_result,
        summary=summary_result, qa=qa_results, frames=frames_result,
    )

    pdf_path = _output_dir() / _safe_filename(meta.title or video_id)
    try:
        build_pdf(
            blocks, pdf_path,
            meta=PdfMeta(
                title=meta.title or "YouTube Video",
                subtitle=meta.channel,
                extra_meta_lines=[f"Mode: {mode}"],
            ),
        )
    except PdfBuildError as exc:
        return f"Could not build the PDF: {exc}"

    # Build result message
    pieces: list[str] = []
    if mode in ("summary", "full") and summary_result.ok:
        pieces.append("summary")
    if mode in ("qa", "full") and any(r.ok for _, r in qa_results):
        pieces.append(f"{sum(1 for _, r in qa_results if r.ok)} answered questions")
    if mode in ("screenshots", "full") and frames_result.ok:
        pieces.append(f"{len(frames_result.frame_paths)} screenshot(s)")
    if mode in ("subtitles", "summary", "qa", "full") and transcript_result.ok:
        pieces.append("full transcript")

    descriptor = ", ".join(pieces) if pieces else "metadata only"
    return f"Built a PDF for '{meta.title or video_id}' with {descriptor}. __FILE_PATH__={pdf_path}"


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------


@tool
def video_to_pdf(
    url_or_query: str,
    mode: str = "full",
    questions: str = "",
    n_frames: int = _DEFAULT_FRAMES,
) -> str:
    """Build a professional PDF from a YouTube video.

    Args:
        url_or_query: A YouTube URL OR a free-text search query.
        mode: ``full`` (default), ``summary``, ``qa``, ``screenshots``, or ``subtitles``.
        questions: Newline- or semicolon-separated questions for ``qa`` or ``full`` mode.
        n_frames: Number of evenly-spaced screenshots (default 8, max 24).

    Returns a message with ``__FILE_PATH__=/path/to/file.pdf`` on success.
    """
    return run_video_to_pdf(
        url_or_query=url_or_query,
        mode=mode if mode in _VALID_MODES else "full",
        questions=questions,
        n_frames=int(n_frames or _DEFAULT_FRAMES),
    )


@tool
def video_qa(url_or_query: str, questions: str) -> str:
    """Answer questions about a YouTube video using its transcript.

    Args:
        url_or_query: A YouTube URL OR a search query.
        questions: One or more questions, separated by newlines or semicolons.
    """
    return run_video_to_pdf(url_or_query=url_or_query, mode="qa", questions=questions, n_frames=0)


@tool
def video_screenshots(url_or_query: str, n_frames: int = _DEFAULT_FRAMES) -> str:
    """Extract evenly-spaced screenshots from a YouTube video into a PDF.

    Args:
        url_or_query: A YouTube URL OR a search query.
        n_frames: Number of screenshots (1-24, default 8).
    """
    return run_video_to_pdf(
        url_or_query=url_or_query, mode="screenshots", questions="",
        n_frames=int(n_frames or _DEFAULT_FRAMES),
    )


def build_video_tools() -> list[Any]:
    """Return the LangChain tool list for the video module."""
    return [video_to_pdf, video_qa, video_screenshots]
