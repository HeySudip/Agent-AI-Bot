"""Shared PDF builder used by every tool that produces a downloadable PDF.

The builder accepts a structured list of *blocks* and renders them with
ReportLab. Blocks can be headings, paragraphs, bullet lists, key/value
tables, code, horizontal rules, page breaks, or images. Keeping the
input data structured (rather than letting each tool render its own
ReportLab story) means we get consistent typography, page numbers, and
metadata across every PDF the bot produces.

The function ``build_pdf`` is the only public entry point.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PdfBuildError",
    "PdfBlock",
    "Heading",
    "Paragraph",
    "Bullets",
    "KeyValue",
    "Code",
    "Image",
    "Rule",
    "PageBreak",
    "build_pdf",
]


class PdfBuildError(RuntimeError):
    """Raised when the PDF backend (ReportLab) is unavailable or fails."""


# --- Block types ------------------------------------------------------------


@dataclass
class PdfBlock:
    """Marker base class so static type checkers can spot mismatches."""


@dataclass
class Heading(PdfBlock):
    text: str
    level: int = 2  # 1 = h1, 2 = h2, 3 = h3


@dataclass
class Paragraph(PdfBlock):
    text: str


@dataclass
class Bullets(PdfBlock):
    items: list[str]


@dataclass
class KeyValue(PdfBlock):
    """Two-column "label: value" rows. Useful for video metadata, etc."""

    rows: list[tuple[str, str]]


@dataclass
class Code(PdfBlock):
    text: str
    language: str = ""


@dataclass
class Image(PdfBlock):
    path: str
    caption: str = ""
    max_width_cm: float = 14.0
    max_height_cm: float = 10.0


@dataclass
class Rule(PdfBlock):
    pass


@dataclass
class PageBreak(PdfBlock):
    pass


# --- Builder ---------------------------------------------------------------


@dataclass
class PdfMeta:
    title: str
    subtitle: str = ""
    author: str = "Agent AI Bot"
    extra_meta_lines: list[str] = field(default_factory=list)


def build_pdf(
    blocks: Iterable[PdfBlock],
    output_path: str | Path,
    *,
    meta: PdfMeta,
) -> str:
    """Render ``blocks`` to a PDF file and return the resolved path.

    Args:
        blocks: Ordered iterable of :class:`PdfBlock` instances.
        output_path: Destination file path. Parent directories are created.
        meta: Title block and document metadata.

    Returns:
        The absolute path of the written file as a string.

    Raises:
        PdfBuildError: If ReportLab is missing or rendering fails.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            ListFlowable,
            ListItem,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.platypus import (
            Image as RLImage,
        )
        from reportlab.platypus import (
            PageBreak as RLPageBreak,
        )
        from reportlab.platypus import (
            Paragraph as RLParagraph,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise PdfBuildError(
            "reportlab is not installed. Install it with: pip install reportlab"
        ) from exc

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#444444"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=15,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=18,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1f3a5c"),
        spaceBefore=14,
        spaceAfter=4,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=colors.HexColor("#2a4d7a"),
        spaceBefore=10,
        spaceAfter=3,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=4,
        textColor=colors.HexColor("#2d2d2d"),
    )
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1a1a1a"),
        backColor=colors.HexColor("#f5f5f5"),
        borderPadding=4,
        leftIndent=4,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Italic"],
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    heading_styles = {1: h1, 2: h2, 3: h3}

    story: list[Any] = []
    story.append(RLParagraph(_escape(meta.title), title_style))
    if meta.subtitle:
        story.append(RLParagraph(_escape(meta.subtitle), subtitle_style))

    meta_lines = [f"Generated by {meta.author} · {time.strftime('%d %B %Y')}"]
    meta_lines.extend(meta.extra_meta_lines)
    story.append(RLParagraph(" · ".join(_escape(line) for line in meta_lines), meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 10))

    for block in blocks:
        if isinstance(block, Heading):
            level = max(1, min(3, block.level))
            story.append(RLParagraph(_escape(block.text), heading_styles[level]))

        elif isinstance(block, Paragraph):
            for para in _split_paragraphs(block.text):
                story.append(RLParagraph(_escape(para), body))
                story.append(Spacer(1, 3))

        elif isinstance(block, Bullets):
            items = [
                ListItem(RLParagraph(_escape(item), body), leftIndent=10)
                for item in block.items
                if item.strip()
            ]
            if items:
                story.append(ListFlowable(items, bulletType="bullet", leftIndent=14))
                story.append(Spacer(1, 4))

        elif isinstance(block, KeyValue):
            data = [
                [RLParagraph(f"<b>{_escape(k)}</b>", body), RLParagraph(_escape(v), body)]
                for k, v in block.rows
                if k or v
            ]
            if data:
                table = Table(data, colWidths=[4 * cm, 13 * cm])
                table.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            (
                                "LINEBELOW",
                                (0, 0),
                                (-1, -1),
                                0.25,
                                colors.HexColor("#e5e5e5"),
                            ),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 6))

        elif isinstance(block, Code):
            text = _escape(block.text).replace("\n", "<br/>")
            story.append(RLParagraph(text, code_style))
            story.append(Spacer(1, 4))

        elif isinstance(block, Image):
            try:
                img = RLImage(
                    block.path,
                    width=block.max_width_cm * cm,
                    height=block.max_height_cm * cm,
                    kind="proportional",
                )
                story.append(img)
                if block.caption:
                    story.append(RLParagraph(_escape(block.caption), caption_style))
                else:
                    story.append(Spacer(1, 4))
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", block.path, exc)
                story.append(
                    RLParagraph(
                        _escape(f"[image unavailable: {block.caption or block.path}]"),
                        caption_style,
                    )
                )

        elif isinstance(block, Rule):
            story.append(
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"))
            )
            story.append(Spacer(1, 4))

        elif isinstance(block, PageBreak):
            story.append(RLPageBreak())

        else:
            logger.debug("Ignoring unknown block type: %s", type(block).__name__)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=meta.title,
        author=meta.author,
    )

    try:
        doc.build(story, onLaterPages=_page_footer, onFirstPage=_page_footer)
    except Exception as exc:
        raise PdfBuildError(f"PDF render failed: {exc}") from exc

    return str(out)


def _page_footer(canvas: Any, doc: Any) -> None:
    """Draw a small page-number footer on every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillGray(0.5)
    canvas.drawRightString(doc.pagesize[0] - 2 * 28.35, 1 * 28.35, f"Page {doc.page}")
    canvas.restoreState()


def _escape(text: str) -> str:
    """Escape ReportLab-XML reserved characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .encode("utf-8", "replace")
        .decode("utf-8")
    )


def _split_paragraphs(text: str) -> list[str]:
    """Split a body string on blank lines so paragraphs render cleanly."""
    if not text:
        return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]
