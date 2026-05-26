"""Tests for the pure-Python parts of the video tool.

Network, yt-dlp, ffmpeg, and Gemini are all mocked. The tests verify URL
parsing, transcript fallback wiring, mode dispatch, and that the
orchestrator emits the ``__FILE_PATH__=`` sentinel on success.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Stub heavy / network deps before we import anything from `tools`.
if "langchain" not in sys.modules:
    pkg = types.ModuleType("langchain")
    sub = types.ModuleType("langchain.tools")

    def _identity(fn):
        return fn

    sub.tool = _identity  # type: ignore[attr-defined]
    pkg.tools = sub  # type: ignore[attr-defined]
    sys.modules["langchain"] = pkg
    sys.modules["langchain.tools"] = sub

if "config" not in sys.modules:
    cfg = types.ModuleType("config")
    cfg.load_config = lambda: {}  # type: ignore[attr-defined]
    cfg.save_config = lambda _c: True  # type: ignore[attr-defined]
    cfg.set_key = lambda _k, _v: True  # type: ignore[attr-defined]
    cfg.get_key = lambda _k, default=None: default  # type: ignore[attr-defined]
    sys.modules["config"] = cfg

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _DummyResp:  # noqa: D401 — minimal stand-in for requests.Response
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {}

    def _noop(*_a: Any, **_kw: Any) -> _DummyResp:  # pragma: no cover
        return _DummyResp()

    class _DummyRequestException(Exception):
        pass

    requests_stub.get = _noop  # type: ignore[attr-defined]
    requests_stub.post = _noop  # type: ignore[attr-defined]
    requests_stub.RequestException = _DummyRequestException  # type: ignore[attr-defined]
    sys.modules["requests"] = requests_stub


def _load(module_name: str, relative_path: str):
    """Load a single module from the project root without importing the package."""
    full_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pdf_builder = _load("pdf_builder_under_test", "tools/pdf_builder.py")
# `video.py` does `from .pdf_builder import ...` and `from safety.ssrf_guard import ...`
# Those imports work only inside the project. Use a thin import shim that pre-binds
# the `tools.pdf_builder` and `tools` package so the relative import succeeds.
if "tools" not in sys.modules:
    tools_pkg = types.ModuleType("tools")
    tools_pkg.__path__ = [str(ROOT / "tools")]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_pkg
sys.modules.setdefault("tools.pdf_builder", _pdf_builder)

_video = _load("tools.video", "tools/video.py")
extract_video_id = _video.extract_video_id
canonical_url = _video.canonical_url
_format_duration = _video._format_duration
_parse_question_list = _video._parse_question_list
TranscriptResult = _video.TranscriptResult
LLMResult = _video.LLMResult
FramesResult = _video.FramesResult
VideoMetadata = _video.VideoMetadata


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/abcDEFghijK", "abcDEFghijK"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ],
    )
    def test_known_url_shapes(self, url: str, expected: str) -> None:
        assert extract_video_id(url) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "not a url",
            "https://example.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/feed/trending",
        ],
    )
    def test_non_video_inputs(self, value: str) -> None:
        assert extract_video_id(value) is None

    def test_canonical_url(self) -> None:
        assert canonical_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestDuration:
    def test_zero(self) -> None:
        assert _format_duration(0) == ""

    def test_seconds_only(self) -> None:
        assert _format_duration(45) == "0m 45s"

    def test_minutes(self) -> None:
        assert _format_duration(125) == "2m 05s"

    def test_hours(self) -> None:
        assert _format_duration(3661) == "1h 01m 01s"


class TestQuestionParser:
    def test_newline_separated(self) -> None:
        out = _parse_question_list(
            "What did the speaker say about RAG?\nWhat is their main critique?"
        )
        assert len(out) == 2
        assert out[0].endswith("RAG?")

    def test_semicolon_separated(self) -> None:
        out = _parse_question_list("Why is sky blue?; what is the weather?")
        assert len(out) == 2

    def test_caps_at_5(self) -> None:
        raw = "\n".join(f"Question number {i}?" for i in range(10))
        assert len(_parse_question_list(raw)) == 5

    def test_empty(self) -> None:
        assert _parse_question_list("") == []

    def test_strips_bullets(self) -> None:
        out = _parse_question_list("- First question?\n• Second question?")
        assert all(not q.startswith(("-", "•")) for q in out)


# ---------------------------------------------------------------------------
# Orchestration / mode dispatch
# ---------------------------------------------------------------------------


class TestRunVideoToPdf:
    def _patch_external(
        self,
        *,
        meta: VideoMetadata,
        transcript: TranscriptResult,
        summary: LLMResult,
        frames: FramesResult,
        qa_results: dict[str, LLMResult] | None = None,
    ):
        qa_results = qa_results or {}
        return [
            patch.object(_video, "fetch_metadata", return_value=meta),
            patch.object(_video, "fetch_transcript", return_value=transcript),
            patch.object(_video, "summarize_transcript", return_value=summary),
            patch.object(
                _video,
                "extract_frames",
                return_value=frames,
            ),
            patch.object(
                _video,
                "answer_question",
                side_effect=lambda _meta, _tx, q: qa_results.get(q, LLMResult(text=f"Answer to: {q}")),
            ),
            patch.object(_video, "build_pdf", return_value="/tmp/fake.pdf"),
        ]

    def _meta(self) -> VideoMetadata:
        return VideoMetadata(
            video_id="dQw4w9WgXcQ",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="Sample",
            channel="Author",
            duration_s=120,
            duration_str="2m 00s",
        )

    def test_invalid_mode_returns_error(self) -> None:
        out = _video.run_video_to_pdf("https://youtu.be/dQw4w9WgXcQ", mode="bogus")
        assert "Invalid mode" in out

    def test_unparseable_url(self) -> None:
        with patch.object(_video, "_resolve_query_to_url", return_value=None):
            out = _video.run_video_to_pdf("not a youtube thing")
            assert "Could not find a YouTube video" in out

    def test_summary_mode_emits_file_path(self) -> None:
        meta = self._meta()
        with patch.object(_video, "fetch_metadata", return_value=meta), \
             patch.object(_video, "fetch_transcript", return_value=TranscriptResult(text="hello world", language="en")), \
             patch.object(_video, "summarize_transcript", return_value=LLMResult(text="## Overview\nA short overview.")), \
             patch.object(_video, "build_pdf", return_value="/tmp/fake.pdf"):
            out = _video.run_video_to_pdf(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ", mode="summary"
            )
            assert "__FILE_PATH__=" in out
            assert "summary" in out

    def test_qa_mode_with_no_questions(self) -> None:
        meta = self._meta()
        with patch.object(_video, "fetch_metadata", return_value=meta), \
             patch.object(_video, "fetch_transcript", return_value=TranscriptResult(text="hi")):
            out = _video.run_video_to_pdf(
                "https://youtu.be/dQw4w9WgXcQ", mode="qa", questions=""
            )
            assert "provide one or more questions" in out

    def test_qa_mode_with_questions(self) -> None:
        meta = self._meta()
        with patch.object(_video, "fetch_metadata", return_value=meta), \
             patch.object(_video, "fetch_transcript", return_value=TranscriptResult(text="hi", language="en")), \
             patch.object(_video, "answer_question", return_value=LLMResult(text="A.")), \
             patch.object(_video, "build_pdf", return_value="/tmp/fake.pdf"):
            out = _video.run_video_to_pdf(
                "https://youtu.be/dQw4w9WgXcQ",
                mode="qa",
                questions="Q1?\nQ2?",
            )
            assert "__FILE_PATH__=" in out
            assert "answered" in out

    def test_screenshots_mode_no_transcript_fetch(self, tmp_path: pathlib.Path) -> None:
        meta = self._meta()
        # Real on-disk file so the orchestrator's "copy to persistent dir" step works.
        fake_frame = tmp_path / "f1.jpg"
        fake_frame.write_bytes(b"\xff\xd8\xff\xe0fake jpeg data")
        with patch.object(_video, "fetch_metadata", return_value=meta), \
             patch.object(_video, "fetch_transcript") as fake_transcript, \
             patch.object(
                 _video,
                 "extract_frames",
                 return_value=FramesResult(frame_paths=[str(fake_frame)], timestamps_s=[10.0]),
             ), \
             patch.object(_video, "build_pdf", return_value="/tmp/fake.pdf"):
            out = _video.run_video_to_pdf(
                "https://youtu.be/dQw4w9WgXcQ", mode="screenshots", n_frames=2
            )
            fake_transcript.assert_not_called()
            assert "screenshot" in out
            assert "__FILE_PATH__=" in out

    def test_subtitles_mode_uses_transcript_only(self) -> None:
        meta = self._meta()
        with patch.object(_video, "fetch_metadata", return_value=meta), \
             patch.object(_video, "fetch_transcript", return_value=TranscriptResult(text="hi", language="en")), \
             patch.object(_video, "extract_frames") as fake_frames, \
             patch.object(_video, "build_pdf", return_value="/tmp/fake.pdf"):
            out = _video.run_video_to_pdf(
                "https://youtu.be/dQw4w9WgXcQ", mode="subtitles"
            )
            fake_frames.assert_not_called()
            assert "transcript" in out


# ---------------------------------------------------------------------------
# PDF builder smoke test
# ---------------------------------------------------------------------------


class TestPdfBuilder:
    def test_smoke_render(self, tmp_path: pathlib.Path) -> None:
        reportlab = pytest.importorskip("reportlab")  # noqa: F841 - import probe
        out = tmp_path / "doc.pdf"
        blocks = [
            _pdf_builder.Heading("Hello", level=1),
            _pdf_builder.Paragraph("This is a paragraph with some <special> chars & ampersands."),
            _pdf_builder.Bullets(["one", "two", "three"]),
            _pdf_builder.Code("print('hi')\nprint('bye')", language="python"),
            _pdf_builder.Rule(),
            _pdf_builder.KeyValue([("Title", "Sample"), ("URL", "https://example.com/")]),
        ]
        _pdf_builder.build_pdf(
            blocks, out, meta=_pdf_builder.PdfMeta(title="Smoke", subtitle="test")
        )
        assert out.exists()
        assert out.stat().st_size > 800  # Non-trivial size
