"""Tests for the LLM response sanitiser.

The "real-world" test reproduces the exact broken WBJEE 2026 response
pasted by the user, ensuring the sanitiser turns it into something
sensible.
"""

from __future__ import annotations

import pytest

from utils.response_sanitizer import (
    FILE_PATH_TAG_RE,
    sanitize_response,
)

# ---------------------------------------------------------------------------
# Chain-of-thought leak removal
# ---------------------------------------------------------------------------


class TestChainOfThoughtRemoval:
    def test_strips_step_headers_when_final_answer_marker_present(self) -> None:
        text = (
            "## Step 1: Understand the Request\n"
            "The user requested a WBJEE 2026 answer key PDF.\n\n"
            "## Step 2: Decide What To Do\n"
            "The exam has not happened.\n\n"
            "The final answer is:\n"
            "I can't find a WBJEE 2026 answer key because the exam hasn't happened yet."
        )
        out = sanitize_response(text)
        assert "Step 1" not in out
        assert "Step 2" not in out
        assert "final answer is" not in out.lower()
        assert "WBJEE 2026 answer key because" in out

    def test_drops_internal_section_headers(self) -> None:
        text = (
            "## Understand the Request\n"
            "The user wants something.\n\n"
            "## Provide a Relevant Response\n"
            "Here is the actual answer to your question."
        )
        out = sanitize_response(text)
        assert "Understand the Request" not in out
        assert "Provide a Relevant Response" not in out
        assert "Here is the actual answer" in out

    def test_strips_research_and_create_pdf_section_header(self) -> None:
        text = (
            "## Research and Create PDF\n"
            "I can't provide a WBJEE 2026 answer key as the exam hasn't occurred yet."
        )
        out = sanitize_response(text)
        assert "## Research and Create PDF" not in out
        assert "exam hasn't occurred yet" in out

    def test_plain_response_unchanged(self) -> None:
        text = "Sure! 17.5% of 248 is 43.4."
        assert sanitize_response(text) == text


# ---------------------------------------------------------------------------
# Hallucinated tool calls
# ---------------------------------------------------------------------------


class TestHallucinatedToolCalls:
    def test_strips_trailing_research_and_create_pdf_call(self) -> None:
        text = (
            "I'll do that for you.\n\n"
            'research_and_create_pdf(query="WBJEE 2026 answer key")'
        )
        out = sanitize_response(text)
        assert "research_and_create_pdf(" not in out
        assert "I'll do that for you" in out

    def test_strips_video_to_pdf_call_anywhere(self) -> None:
        text = (
            "Here is the explanation.\n"
            'video_to_pdf(url_or_query="https://youtu.be/abc", mode="summary")\n'
            "Hope that helps."
        )
        out = sanitize_response(text)
        assert "video_to_pdf(" not in out
        assert "Hope that helps" in out

    def test_does_not_strip_legitimate_function_mentions(self) -> None:
        text = "You can call calculate(2+2) in Python — that returns 4."
        out = sanitize_response(text)
        # `calculate` is in the allow-list, but the line is plain prose.
        # Our regex requires the call to BE the line (after optional bullet),
        # so a sentence containing it should be kept.
        assert "calculate(2+2)" in out

    def test_strips_call_inside_code_fence_marker(self) -> None:
        text = (
            "Sure thing.\n"
            '`generate_text_to_pdf(text="hi", filename="note")`'
        )
        out = sanitize_response(text)
        assert "generate_text_to_pdf(" not in out
        assert "Sure thing." in out


# ---------------------------------------------------------------------------
# Unfulfilled file claims
# ---------------------------------------------------------------------------


class TestUnfulfilledFileClaims:
    def test_strips_heres_your_pdf_when_no_file_tag(self) -> None:
        text = "Here's your PDF!"
        out = sanitize_response(text)
        assert "Here's your PDF" not in out

    def test_keeps_heres_your_pdf_when_file_tag_present(self) -> None:
        text = "Here's your PDF! __FILE_PATH__=/tmp/answer.pdf"
        out = sanitize_response(text)
        assert "Here's your PDF" in out
        assert "__FILE_PATH__=/tmp/answer.pdf" in out

    def test_strips_your_file_is_ready_no_tag(self) -> None:
        text = "Your file is ready."
        out = sanitize_response(text)
        assert "file is ready" not in out.lower()

    def test_strips_ive_generated_the_document_no_tag(self) -> None:
        text = "I've generated the document for you."
        out = sanitize_response(text)
        assert "generated the document" not in out.lower()

    def test_full_message_replaced_by_fallback_when_only_false_claim(self) -> None:
        text = "Here's your PDF!"
        out = sanitize_response(text)
        # Sanitising a message that was nothing but a false claim should
        # produce the safe fallback rather than an empty string.
        assert out
        assert "Here's your PDF" not in out


# ---------------------------------------------------------------------------
# End-to-end: the exact broken response from the user's bug report
# ---------------------------------------------------------------------------


class TestRealWorldBugReport:
    BROKEN = """\
## Step 1: Understand the Request
The user requested a WBJEE 2026 answer key PDF. However, since the exam year 2026 is in the future, the exam has not yet taken place, and therefore, the answer key has not been released.

## Step 2: Determine the Appropriate Action
Given that the WBJEE 2026 exam has not occurred yet, I should not provide any fictional or incorrect information. Instead, I should offer to help the user find resources or information that might be useful.

## Step 3: Provide a Relevant Response
I can suggest some possible resources where the user might find the WBJEE 2026 answer key or related information in the future:
- Official WBJEE website: The most reliable source for the answer key would be the official WBJEE website.
- Educational portals: Websites like Collegedekho, AglaSem, or Careers360 often provide answer keys and solutions for various exams.

## Step 4: Offer Assistance
I can offer to assist the user in finding the answer key or solutions for the WBJEE 2026 exam once it is available.

The final answer is:
## Research and Create PDF
I can't provide a WBJEE 2026 answer key PDF as the exam hasn't occurred yet. Answer keys are typically released after the examination.

If you're looking for information on a past WBJEE exam, please let me know the specific year, and I'll do my best to find it for you.

Here's your PDF!
 research_and_create_pdf(query="WBJEE 2026 answer key")"""

    def test_no_chain_of_thought_in_output(self) -> None:
        out = sanitize_response(self.BROKEN)
        for marker in (
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "## Understand the Request",
            "## Determine the Appropriate Action",
            "## Provide a Relevant Response",
            "## Offer Assistance",
            "## Research and Create PDF",
            "The final answer is",
        ):
            assert marker not in out, f"sanitiser left behind {marker!r}"

    def test_no_hallucinated_tool_call_in_output(self) -> None:
        out = sanitize_response(self.BROKEN)
        assert "research_and_create_pdf(" not in out

    def test_no_false_pdf_claim_in_output(self) -> None:
        out = sanitize_response(self.BROKEN)
        # The original said "Here's your PDF!" but no __FILE_PATH__ tag.
        # The sanitiser must have removed that claim.
        assert "Here's your PDF" not in out
        assert not FILE_PATH_TAG_RE.search(out)

    def test_keeps_the_actual_answer(self) -> None:
        out = sanitize_response(self.BROKEN)
        assert "WBJEE 2026 answer key" in out
        assert "exam hasn't occurred yet" in out


# ---------------------------------------------------------------------------
# Idempotence and odd inputs
# ---------------------------------------------------------------------------


class TestIdempotence:
    @pytest.mark.parametrize(
        "text",
        [
            "Plain answer.",
            "## Step 1: Plan\nDo X.\n\nThe final answer is:\nUse Y.",
            "Here's your PDF! __FILE_PATH__=/tmp/x.pdf",
        ],
    )
    def test_running_twice_changes_nothing(self, text: str) -> None:
        once = sanitize_response(text)
        twice = sanitize_response(once)
        assert once == twice


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert sanitize_response("") == ""

    def test_none_returns_empty(self) -> None:
        assert sanitize_response(None) == ""  # type: ignore[arg-type]

    def test_whitespace_only(self) -> None:
        assert sanitize_response("   \n\n  ") == "   \n\n  "

    def test_non_string(self) -> None:
        assert sanitize_response(42) == 42  # type: ignore[arg-type]
