"""Clean up LLM artifacts before showing the response to the user.

The agent occasionally returns text that violates the system prompt:

1. **Chain-of-thought leak.** The model dumps its planning ("Step 1: …",
   "Step 2: …", "The final answer is: …") instead of just the answer.
2. **Fake tool calls as text.** The model writes ``research_and_create_pdf(query="…")``
   as plain text instead of actually invoking the tool.
3. **False file claims.** The model says "Here's your PDF!" without ever
   producing a ``__FILE_PATH__=…`` tag, so the bot replies with no file
   attached.

This module surgically removes those artifacts. It runs after
:func:`agent._extract_text` and before the response is sent back to the
Telegram handler. It only deletes things that match well-defined
patterns; ordinary user-facing prose is left alone.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

__all__ = ["sanitize_response", "FILE_PATH_TAG_RE"]

logger = logging.getLogger(__name__)

FILE_PATH_TAG_RE: re.Pattern[str] = re.compile(r"__FILE_PATH__=\S+")

# Tool names the model might hallucinate as text.
_KNOWN_TOOL_NAMES: tuple[str, ...] = (
    "research_and_create_pdf",
    "generate_text_to_pdf",
    "youtube_video_to_pdf",
    "video_to_pdf",
    "video_qa",
    "video_screenshots",
    "extract_youtube_to_pdf",
    "search_and_extract_youtube_to_pdf",
    "fetch_and_summarize_url",
    "get_page_title_and_description",
    "extract_links_from_url",
    "search_web",
    "calculate",
    "get_current_datetime",
    "convert_units",
    "encode_decode",
    "generate_text",
    "format_json",
    "compare_texts",
    "format_and_analyze_code",
    "save_api_key",
)

_TOOL_CALL_AS_TEXT_RE: re.Pattern[str] = re.compile(
    r"""
    (?:^|\n)               # start of line
    [\s>•\-*]*             # optional leading whitespace / bullets / quotes
    (?:`{{0,3}}|\*{{0,2}})     # optional code-fence or bold markers
    (?P<name>{})           # one of the known tool names
    \s*\([^()\n]{{0,500}}\)  # one-line argument list
    \s*`{{0,3}}\s*\.?        # optional trailing fence and period
    (?=\n|$)               # end of line
    """.format("|".join(re.escape(n) for n in _KNOWN_TOOL_NAMES)),
    re.VERBOSE | re.IGNORECASE,
)

_FAKE_FILE_CLAIM_RE: re.Pattern[str] = re.compile(
    r"""
    (?:^|(?<=[\s>•\-*]))
    (?:here\s*(?:'?s|\s+is)\s+your\s+(?:pdf|file|document|answer\s+key|notes|paper)!?
       |your\s+(?:pdf|file|document)\s+is\s+(?:ready|attached|generated)!?
       |i\s*['']?ve\s+(?:created|generated|made)\s+(?:the|your|a)\s+(?:pdf|file|document)
       |the\s+(?:pdf|file|document)\s+is\s+(?:ready|attached|generated)!?
    )
    [!.]?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_COT_HEADERS_RE: re.Pattern[str] = re.compile(
    r"""^[ \t]*
    (?:\#{1,4}\s*)?           # optional markdown heading
    (?:Step\s+\d+\s*[:.]      # "Step 1:", "Step 2."
       |Final\s+Answer\s*[:.]?
       |The\s+final\s+answer\s+is\s*[:.]?
       |Reasoning\s*[:.]?
       |Plan\s*[:.]?
       |Thought\s*[:.]?
       |Action\s*[:.]?
       |Observation\s*[:.]?
    )
    [ \t]*$
    """,
    re.VERBOSE | re.MULTILINE | re.IGNORECASE,
)

_PLAN_HEADERS_RE: re.Pattern[str] = re.compile(
    r"""^[ \t]*\#{1,4}\s+
    (?:Understand\s+the\s+Request
       |Determine\s+the\s+(?:Appropriate\s+)?Action
       |Provide\s+a\s+Relevant\s+Response
       |Offer\s+Assistance
       |Research\s+and\s+Create\s+PDF
    )
    [ \t]*$
    """,
    re.VERBOSE | re.MULTILINE | re.IGNORECASE,
)


def _has_cot_structure(text: str) -> bool:
    """Return True when the text looks like a multi-step planning monologue."""
    step_count = len(
        re.findall(r"^\s*\#{0,4}\s*Step\s+\d+", text, re.MULTILINE | re.IGNORECASE)
    )
    return step_count >= 2 or bool(
        re.search(r"^\s*The final answer is\b", text, re.MULTILINE | re.IGNORECASE)
    )


def _extract_after_final_answer(text: str) -> str | None:
    """Return everything after a 'final answer' / 'final response' marker."""
    marker = re.search(
        r"(?:^|\n)\s*(?:\#{0,4}\s*)?"
        r"(?:The\s+final\s+answer\s+is|Final\s+answer|Final\s+response)\s*[:.]?\s*\n+",
        text,
        re.IGNORECASE,
    )
    if marker:
        tail = text[marker.end() :].strip()
        return tail or None
    return None


def _strip_cot_blocks(text: str) -> str:
    """Remove planning headers and the body of visible Step blocks."""
    pattern = re.compile(
        r"""
        ^[ \t]*(?:\#{1,4}\s*)?
        Step\s+\d+\s*[:.][^\n]*\n
        (?:.+?)?
        (?=
            ^[ \t]*(?:\#{1,4}\s*)?Step\s+\d+
            |^[ \t]*(?:\#{1,4}\s*)?(?:Final\s+answer|The\s+final\s+answer)
            |^[ \t]*\#{1,4}\s+\w
            |\n[ \t]*\n
            |\Z
        )
        """,
        re.VERBOSE | re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    cleaned = pattern.sub("", text)
    cleaned = _COT_HEADERS_RE.sub("", cleaned)
    cleaned = _PLAN_HEADERS_RE.sub("", cleaned)
    return cleaned


def _strip_hallucinated_tool_calls(text: str) -> tuple[str, list[str]]:
    """Remove plaintext that looks like a tool invocation.

    Returns the cleaned text plus the list of tool names found.
    """
    found: list[str] = []

    def _replace(m: re.Match[str]) -> str:
        found.append(m.group("name"))
        return ""

    cleaned = _TOOL_CALL_AS_TEXT_RE.sub(_replace, text)
    return cleaned, found


def _strip_unfulfilled_file_claims(text: str) -> str:
    """Drop 'Here's your PDF' lines when no real file path is present."""
    if FILE_PATH_TAG_RE.search(text):
        return text
    return _FAKE_FILE_CLAIM_RE.sub("", text)


def _collapse_blank_runs(text: str) -> str:
    """Collapse runs of three-or-more newlines down to a paragraph break."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sanitize_response(text: str) -> str:
    """Apply every sanitiser in order. Safe to call on any string.

    The function is intentionally idempotent: calling it twice produces
    the same result as calling it once.
    """
    if not isinstance(text, str) or not text.strip():
        return text or ""

    # 1. If the model dumped a multi-step plan, prefer the post-"Final answer" tail.
    if _has_cot_structure(text):
        tail = _extract_after_final_answer(text)
        if tail and len(tail.strip()) >= 20:
            text = tail
        else:
            text = _strip_cot_blocks(text)

    # 2. Always strip stray plan headers, even outside CoT structure.
    text = _PLAN_HEADERS_RE.sub("", text)

    # 3. Remove plaintext tool-call hallucinations.
    text, hallucinated = _strip_hallucinated_tool_calls(text)
    if hallucinated:
        logger.info(
            "Stripped hallucinated tool-call text from LLM response: %s",
            ", ".join(sorted(set(hallucinated))),
        )

    # 4. Drop false file-ready claims.
    text = _strip_unfulfilled_file_claims(text)

    # 5. Tidy up whitespace.
    text = _collapse_blank_runs(text)

    if not text.strip():
        logger.warning("LLM response was empty after sanitisation; using fallback.")
        return (
            "I wasn't able to put together a proper answer this time. "
            "Try asking again with a bit more detail."
        )
    return text


def iter_known_tool_names() -> Iterable[str]:
    """Expose the tool-name list for tests and diagnostics."""
    return _KNOWN_TOOL_NAMES
