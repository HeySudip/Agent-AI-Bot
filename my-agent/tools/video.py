"""Professional video → PDF tool.

Modes
=====
- ``subtitles``   — fetch the transcript only (no LLM, no frames).
- ``screenshots`` — extract evenly-spaced frames from the video.
- ``summary``     — LLM-generated structured summary of the transcript.
- ``qa``          — answer specific question(s) using the transcript.
- ``full``        — metadata + summary + selected screenshots + transcript.

Pipeline
========
1. Resolve the URL / search query → ``video_id``, canonical URL, metadata.
2. Fetch transcript with multi-language fallback.
3. (Optional) Download lowest-quality video via yt-dlp, extract N frames
   via the ffmpeg binary shipped by ``imageio_ffmpeg``.
4. (Optional) Run the transcript through Gemini for summary / Q&A.
5. Render a professional PDF (metadata table, summary, screenshots, full
   transcript) using the shared :mod:`tools.pdf_builder`.

The tool gracefully degrades when optional dependencies are unavailable —
each step records a status flag, and the resulting PDF includes a
"Processing notes" section so the user knows exactly what worked and
what didn't.
"""

from __future__ import annotations

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
from .pdf_builder import (
    Image as PdfImage,
)
from .pdf_builder import (
    Paragraph as PdfParagraph,
)

logger = logging.getLogger(__name__)

VideoMode = Literal["subtitles", "screenshots", "summary", "qa", "full"]
_VALID_MODES: tuple[str, ...] = ("subtitles", "screenshots", "summary", "qa", "full")

_DEFAULT_FRAMES = 8
_MAX_FRAMES = 24
_MIN_FRAMES = 1

_VIDEO_DOWNLOAD_TIMEOUT_S = 180
_FRAME_EXTRACTION_TIMEOUT_S = 90

_TRANSCRIPT_PREFERRED_LANGS = [
    "en",
    "en-US",
    "en-GB",
    "hi",
    "es",
    "fr",
    "de",
    "pt",
    "ru",
    "ja",
    "ko",
    "zh",
    "ar",
    "id",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoMetadata:
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
    text: str = ""
    language: str = ""
    is_generated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


@dataclass
class FramesResult:
    frame_paths: list[str] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.frame_paths)


@dataclass
class LLMResult:
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def extract_video_id(url: str) -> str | None:
    """Return the YouTube video id from a watch / youtu.be / shorts URL."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host in {"youtu.be", "youtu.be."}:
        return parsed.path.lstrip("/").split("/")[0] or None
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
        if parsed.path.startswith("/shorts/"):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 and parts[2] else None
        if parsed.path.startswith("/embed/"):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 and parts[2] else None
    return None


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _format_duration(seconds: int) -> str:
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
    """Search for a YouTube video matching ``query`` and return its URL."""
    candidates: list[dict[str, Any]] = []
    try:
        from ddgs import DDGS  # type: ignore[import-untyped]
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]
        except ImportError:
            DDGS = None  # type: ignore[assignment]

    if DDGS is not None:
        try:
            with DDGS() as d:
                candidates = list(d.text(f"site:youtube.com {query}", max_results=8))
        except Exception as exc:
            logger.info("DDG search failed: %s", exc)

    for hit in candidates:
        href = hit.get("href") or hit.get("url") or ""
        if extract_video_id(href):
            return href
    return None


def fetch_metadata(video_id: str, url: str) -> VideoMetadata:
    """Pull video metadata via yt-dlp, gracefully handling its absence."""
    meta = VideoMetadata(video_id=video_id, url=url, thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
    try:
        import yt_dlp  # type: ignore[import-untyped]
    except ImportError:
        logger.info("yt-dlp not installed; skipping metadata fetch")
        return meta

    opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,
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
    desc = info.get("description") or ""
    meta.description = desc.strip()[:1500]
    return meta


# ---------------------------------------------------------------------------
# Transcript fetching
# ---------------------------------------------------------------------------


def fetch_transcript(video_id: str, preferred_languages: list[str] | None = None) -> TranscriptResult:
    """Fetch a transcript with manual-then-auto, preferred-then-any fallback."""
    languages = preferred_languages or _TRANSCRIPT_PREFERRED_LANGS

    try:
        from youtube_transcript_api import (  # type: ignore[import-untyped]
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )
    except ImportError as exc:
        return TranscriptResult(error=f"youtube_transcript_api not installed: {exc}")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        return TranscriptResult(error="No transcripts are available for this video.")
    except VideoUnavailable:
        return TranscriptResult(error="The video is unavailable or private.")
    except Exception as exc:
        return TranscriptResult(error=f"Failed to list transcripts: {exc}")

    selected = None
    selected_lang = ""
    is_generated = False

    # 1) Manually-created transcript in a preferred language.
    try:
        selected = transcript_list.find_manually_created_transcript(languages)
        selected_lang = selected.language_code
    except Exception:
        selected = None

    # 2) Auto-generated in a preferred language.
    if selected is None:
        try:
            selected = transcript_list.find_generated_transcript(languages)
            selected_lang = selected.language_code
            is_generated = True
        except Exception:
            selected = None

    # 3) Anything in any language, then translate to English if possible.
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

    target_lang = selected_lang
    fetched: list[dict[str, Any]] = []
    try:
        if selected_lang not in languages:
            try:
                translated = selected.translate("en")
                fetched = list(translated.fetch())
                target_lang = "en (translated)"
            except Exception:
                fetched = list(selected.fetch())
        else:
            fetched = list(selected.fetch())
    except Exception as exc:
        return TranscriptResult(error=f"Transcript fetch failed: {exc}")

    text = " ".join(seg.get("text", "").strip() for seg in fetched if seg.get("text"))
    text = re.sub(r"\s+", " ", text).strip()
    return TranscriptResult(text=text, language=target_lang, is_generated=is_generated)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _ffmpeg_binary() -> str | None:
    """Return a path to an ffmpeg binary, preferring the imageio-ffmpeg one."""
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and Path(path).exists():
            return path
    except ImportError:
        pass
    system_ffmpeg = shutil.which("ffmpeg")
    return system_ffmpeg


def _download_video(url: str, work_dir: Path) -> tuple[Path | None, str]:
    """Download the lowest-quality video to ``work_dir``. Return (path, error)."""
    try:
        import yt_dlp  # type: ignore[import-untyped]
    except ImportError:
        return None, "yt-dlp is not installed; cannot download video for screenshots."

    out_template = str(work_dir / "video.%(ext)s")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "worst[ext=mp4]/worst",
        "outtmpl": out_template,
        "noplaylist": True,
        "concurrent_fragment_downloads": 1,
        "socket_timeout": 30,
        "retries": 2,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        return None, f"yt-dlp download failed: {exc}"

    candidates = sorted(work_dir.glob("video.*"))
    return (candidates[0] if candidates else None,
            "" if candidates else "Video file not produced by yt-dlp.")


def _download_thumbnail(url: str, dest: Path, timeout: float = 15.0) -> bool:
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
    """Extract ``n_frames`` evenly spaced frames from the video.

    Falls back to the YouTube thumbnail if yt-dlp / ffmpeg are unavailable.
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

    # Compute capture timestamps.
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
            ffmpeg,
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-loglevel",
            "error",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=False,
                timeout=_FRAME_EXTRACTION_TIMEOUT_S,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
    return FramesResult(frame_paths=frame_paths, timestamps_s=timestamps[: len(frame_paths)])


# ---------------------------------------------------------------------------
# LLM summarization / Q&A
# ---------------------------------------------------------------------------


def _gemini_complete(prompt: str, *, max_output_tokens: int = 2048, temperature: float = 0.2) -> LLMResult:
    """Call the Gemini text API with the user's saved key. Returns LLMResult."""
    try:
        from config import load_config  # local import: bot project module
    except ImportError as exc:
        return LLMResult(error=f"config module not importable: {exc}")

    cfg = load_config()
    api_key = (
        cfg.get("gemini_api_key")
        or (cfg.get("gemini_api_keys") or [None])[0]
        or ""
    )
    if not api_key:
        return LLMResult(error="No Gemini key configured; cannot summarize / answer.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=45)
    except requests.RequestException as exc:
        return LLMResult(error=f"Gemini request failed: {exc}")

    if resp.status_code != 200:
        return LLMResult(error=f"Gemini API HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        text = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
    except Exception as exc:
        return LLMResult(error=f"Gemini response parse failed: {exc}")
    return LLMResult(text=text) if text else LLMResult(error="Gemini returned empty text.")


def summarize_transcript(meta: VideoMetadata, transcript: str) -> LLMResult:
    if not transcript.strip():
        return LLMResult(error="Empty transcript; nothing to summarize.")
    prompt = (
        "You are creating a concise, well-structured summary of a video.\n\n"
        f"Title: {meta.title or '(unknown)'}\n"
        f"Channel: {meta.channel or '(unknown)'}\n"
        f"Duration: {meta.duration_str or '(unknown)'}\n\n"
        "Produce the summary using exactly this Markdown layout:\n\n"
        "## Overview\n"
        "Two or three sentences describing what the video is about.\n\n"
        "## Key points\n"
        "Five to ten bullet points capturing the main ideas, in order.\n\n"
        "## Notable quotes or facts\n"
        "Up to five short quoted lines from the transcript that are worth highlighting.\n\n"
        "## Takeaways\n"
        "Three concrete, actionable takeaways for the viewer.\n\n"
        "Only use facts present in the transcript. Do not invent information.\n\n"
        "TRANSCRIPT:\n"
        f"{transcript[:18000]}"
    )
    return _gemini_complete(prompt, max_output_tokens=2048, temperature=0.2)


def answer_question(meta: VideoMetadata, transcript: str, question: str) -> LLMResult:
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
        "TRANSCRIPT:\n"
        f"{transcript[:18000]}"
    )
    return _gemini_complete(prompt, max_output_tokens=1024, temperature=0.1)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _parse_question_list(raw: str) -> list[str]:
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
    """Chunk transcript text into readable paragraphs."""
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


def run_video_to_pdf(
    url_or_query: str,
    *,
    mode: VideoMode = "full",
    questions: str = "",
    n_frames: int = _DEFAULT_FRAMES,
) -> str:
    """Driver function. Returns a string suitable for the agent to relay."""
    if mode not in _VALID_MODES:
        return (
            f"Invalid mode '{mode}'. Use one of: {', '.join(_VALID_MODES)}."
        )

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

    transcript_result = TranscriptResult()
    if mode in ("subtitles", "summary", "qa", "full"):
        transcript_result = fetch_transcript(video_id)

    questions_list: list[str] = []
    qa_results: list[tuple[str, LLMResult]] = []
    if mode in ("qa", "full"):
        questions_list = _parse_question_list(questions) if questions else []
        if mode == "qa" and not questions_list:
            return "Please provide one or more questions for QA mode."
        for q in questions_list:
            qa_results.append((q, answer_question(meta, transcript_result.text, q)))

    summary_result = LLMResult()
    if mode in ("summary", "full") and transcript_result.ok:
        summary_result = summarize_transcript(meta, transcript_result.text)

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

    blocks = _build_blocks(
        meta=meta,
        mode=mode,
        transcript=transcript_result,
        summary=summary_result,
        qa=qa_results,
        frames=frames_result,
    )

    pdf_path = _output_dir() / _safe_filename(meta.title or video_id)
    try:
        build_pdf(
            blocks,
            pdf_path,
            meta=PdfMeta(
                title=meta.title or "YouTube Video",
                subtitle=meta.channel,
                extra_meta_lines=[f"Mode: {mode}"],
            ),
        )
    except PdfBuildError as exc:
        return f"Could not build the PDF: {exc}"

    pieces = []
    if mode in ("summary", "full") and summary_result.ok:
        pieces.append("summary")
    if mode in ("qa", "full") and any(r.ok for _, r in qa_results):
        pieces.append(f"{sum(1 for _, r in qa_results if r.ok)} answered questions")
    if mode in ("screenshots", "full") and frames_result.ok:
        pieces.append(f"{len(frames_result.frame_paths)} screenshot(s)")
    if mode in ("subtitles", "summary", "qa", "full") and transcript_result.ok:
        pieces.append("full transcript")

    descriptor = ", ".join(pieces) if pieces else "metadata only"
    return (
        f"Built a PDF for '{meta.title or video_id}' with {descriptor}. "
        f"__FILE_PATH__={pdf_path}"
    )


def _build_blocks(
    *,
    meta: VideoMetadata,
    mode: VideoMode,
    transcript: TranscriptResult,
    summary: LLMResult,
    qa: list[tuple[str, LLMResult]],
    frames: FramesResult,
) -> list[Any]:
    blocks: list[Any] = []

    blocks.append(KeyValue(rows=_meta_rows(meta)))
    if meta.description:
        blocks.append(Heading("Description", level=2))
        blocks.append(PdfParagraph(meta.description))
    blocks.append(Rule())

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
            blocks.append(
                PdfParagraph(summary.error or "Summary unavailable for this video.")
            )

    if mode in ("qa", "full") and qa:
        blocks.append(PageBreak())
        blocks.append(Heading("Questions & Answers", level=1))
        for question, result in qa:
            blocks.append(Heading(question, level=2))
            blocks.append(
                PdfParagraph(result.text if result.ok else (result.error or "No answer."))
            )

    if mode in ("screenshots", "full"):
        blocks.append(PageBreak())
        blocks.append(Heading("Screenshots", level=1))
        if frames.ok:
            for i, frame in enumerate(frames.frame_paths, start=1):
                ts = frames.timestamps_s[i - 1] if i - 1 < len(frames.timestamps_s) else 0.0
                caption = (
                    f"Frame {i} · t = {_format_duration(int(ts))}" if ts else f"Frame {i}"
                )
                blocks.append(PdfImage(path=frame, caption=caption))
        else:
            blocks.append(
                PdfParagraph(frames.error or "No screenshots could be extracted.")
            )

    if mode in ("subtitles", "full"):
        blocks.append(PageBreak())
        lang = transcript.language or "unknown"
        kind = "auto-generated" if transcript.is_generated else "manual"
        blocks.append(Heading(f"Transcript ({lang}, {kind})", level=1))
        if transcript.ok:
            blocks.extend(_transcript_blocks(transcript.text))
        else:
            blocks.append(
                PdfParagraph(transcript.error or "Transcript unavailable for this video.")
            )

    notes = _processing_notes(transcript=transcript, summary=summary, frames=frames, qa=qa, mode=mode)
    if notes:
        blocks.append(Rule())
        blocks.append(Heading("Processing notes", level=3))
        blocks.append(Bullets(notes))

    return blocks


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
        mode: ``full`` (default), ``summary``, ``qa``, ``screenshots``, or
            ``subtitles``.
        questions: Newline- or semicolon-separated list of questions to
            answer when ``mode='qa'`` or as part of ``mode='full'``.
        n_frames: Number of evenly-spaced screenshots to extract
            (default 8, max 24).

    Returns either an explanatory message ending with
    ``__FILE_PATH__=/path/to/file.pdf`` or a plain error string when no
    PDF could be produced.

    Use this whenever the user wants:
    - a summary of a YouTube video
    - answers to questions about a YouTube video
    - subtitles / transcript of a YouTube video as a PDF
    - screenshots of a YouTube video
    """
    return run_video_to_pdf(
        url_or_query=url_or_query,
        mode=mode if mode in _VALID_MODES else "full",
        questions=questions,
        n_frames=int(n_frames or _DEFAULT_FRAMES),
    )


@tool
def video_qa(url_or_query: str, questions: str) -> str:
    """Answer one or more questions about a YouTube video using its transcript.

    Args:
        url_or_query: A YouTube URL OR a search query.
        questions: One or more questions, separated by newlines or
            semicolons. Each is answered using only the transcript.

    Returns the standard ``__FILE_PATH__=...`` reply on success.
    """
    return run_video_to_pdf(
        url_or_query=url_or_query, mode="qa", questions=questions, n_frames=0
    )


@tool
def video_screenshots(url_or_query: str, n_frames: int = _DEFAULT_FRAMES) -> str:
    """Extract evenly-spaced screenshots from a YouTube video into a PDF.

    Args:
        url_or_query: A YouTube URL OR a search query.
        n_frames: Number of screenshots to take (1 - 24, default 8).
    """
    return run_video_to_pdf(
        url_or_query=url_or_query,
        mode="screenshots",
        questions="",
        n_frames=int(n_frames or _DEFAULT_FRAMES),
    )


def build_video_tools() -> list[Any]:
    """Return the LangChain tool list for the video module."""
    return [video_to_pdf, video_qa, video_screenshots]
