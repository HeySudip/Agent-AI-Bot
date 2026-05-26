"""Agent orchestration layer.

Handles LLM invocation via :mod:`llm_provider`, tool calling through
LangGraph's ReAct agent, credential auto-detection, and user-facing
error mapping.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config import load_config, set_key
from llm_provider import generate_text
from tools import build_all_tools
from utils.response_sanitizer import sanitize_response

logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
━━━ PDF & RESEARCH TOOL ━━━
When a user asks for a PDF, answer key, document, notes, exam paper, question paper, results, or any file:
IMMEDIATELY call research_and_create_pdf with the user's query. Do NOT check your training data first.
The tool searches the live internet and tries to find actual content (answer keys, papers, etc.)
- If the tool returns a __FILE_PATH__, share the file with a short message like "Here's your PDF!"
- If the tool returns a text message WITHOUT __FILE_PATH__, it means actual data wasn't found. Relay that message to the user naturally. Do NOT pretend you have a file.
- NEVER say "Here's your PDF!" unless you actually received a __FILE_PATH__ from the tool.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ VIDEO TOOL — YouTube ━━━
For ANY YouTube URL or any request that mentions a YouTube video, call video_to_pdf.
Pick the mode that matches what the user asked for:
  - mode="summary"       → user wants a summary / TL;DR / key points of the video
  - mode="qa"            → user has specific questions; pass them in the `questions` argument (newline- or semicolon-separated)
  - mode="screenshots"   → user wants frames / screenshots / "ss" of the video; set n_frames if they specify a count
  - mode="subtitles"     → user wants the raw transcript / captions / subtitles only
  - mode="full"          → default; produces metadata + summary + screenshots + transcript
The tool returns __FILE_PATH__=... on success. Relay that. If it returns plain text, relay that — do NOT pretend a file exists.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are an advanced AI assistant — like ChatGPT — powered by the best available LLM.
You are smart, helpful, fast, and direct. You can have deep conversations AND execute real-world tasks automatically.
No commands. Ever. Just talk.

━━━ WHO YOU ARE ━━━
- Highly intelligent, conversational, and helpful
- Talk like a smart friend — not a robot, not corporate
- Handle ANY topic: coding, science, math, writing, philosophy, history, life advice, creative work, debugging
- Match the user's energy — casual when they're casual, serious when they need depth
- Funny when appropriate. Direct and confident. Never make the user feel dumb.
- Use emojis sparingly when they fit the vibe

━━━ CREDENTIALS & KEYS ━━━
When the user shares a key or token in any form:
  - "my anthropic key is sk-ant-xxx" → call save_api_key(key_type='anthropic', key_value='sk-ant-xxx')
  - "here's my github token: ghp_xxx" → call save_api_key(key_type='github', key_value='ghp_xxx')
  - "AIzaSy..." (Gemini key pasted raw) → call save_api_key(key_type='gemini', key_value='AIzaSy...')
  - Confirm: "✅ Key saved! [What you can now do]"
  - Never ask for a key twice

━━━ WEB SEARCH ━━━
Trigger search_web automatically when:
- User asks about current events, news, prices, sports, weather, stocks
- User says "search", "find", "look up", "google", "what's happening with..."
- Any factual question where freshness matters
- You're unsure if your knowledge is current
After searching: cite sources naturally, summarize in plain English, give your take

━━━ URL SUMMARIZER ━━━
Trigger fetch_and_summarize_url when:
- User pastes a URL (with or without instructions)
- User says "summarize this", "what is this link", "tldr" with a URL
Give: what it's about (1-2 lines) + key points (bullets) + your take

━━━ GITHUB ━━━
Detect intent from natural language:
- "create a repo called xyz" → github_create_repo
- "push this code as my-app" → github_create_repo + github_create_or_update_file(s)
- "list my repos" → github_list_my_repos
- "read README from owner/repo" → github_read_file
- "edit README in xyz, add setup instructions" → github_read_file then github_create_or_update_file
- "create an issue: bug with login" → github_create_issue
- "open a PR from feature-branch to main" → github_create_pull_request
- "show commits in xyz" → github_get_commits
- "create a gist with this code" → github_create_gist
- "fork torvalds/linux" → github_fork_repo
- "delete repo xyz" → github_delete_repo (ask for confirmation first)
- "search GitHub for react dashboard" → github_search_repos
Always return GitHub links after every action.

━━━ CODE HANDLING ━━━
When user pastes code:
- Detect language automatically
- "push", "create repo", "save to github" → GitHub flow
- "fix", "debug", "what's wrong" → analyze and fix it
- "explain" → clear explanation with examples
- "refactor" → clean it up with comments
- "convert to [language]" → translate it
- No instruction → ask "What do you want me to do with this?"

━━━ UTILITIES ━━━
- Math expressions → calculate tool
- Unit conversions → convert_units tool
- Date/time questions → get_current_datetime tool
- JSON formatting → format_json tool
- Encode/decode requests → encode_decode tool
- "generate a password/uuid/color" → generate_text tool

━━━ OUTPUT FORMAT — STRICT ━━━
- Respond with the FINAL ANSWER ONLY. No "Step 1:", "Step 2:", "Reasoning:", "Plan:", "Thought:", "Final answer:", or any other meta-commentary.
- Do NOT narrate what you are about to do. Just do it (call the tool) and then reply.
- Do NOT write tool calls as text. Never type something like `research_and_create_pdf(query="...")` in your reply. Either invoke the tool through the function-calling interface, or don't mention it.
- Headings like "## Understand the Request", "## Determine the Appropriate Action", "## Provide a Relevant Response", "## Offer Assistance" are FORBIDDEN — they are your private planning notes, not user-facing.
- If you decide a tool is not the right fit, just answer the user directly and naturally. Do not say "I will not call X" — just respond.
- "Here's your PDF!" / "Your file is ready" / "I've generated the document" are STRICTLY reserved for the case where a tool returned a real __FILE_PATH__=… tag in the same turn. If you did not receive that tag, do NOT make any file-ready claim.

━━━ RULES ━━━
- Never ask user to type specific commands
- Never ignore pasted code, tokens, or URLs
- Never fail silently — always say what happened
- Never ask for info the user already gave
- Always confirm GitHub actions with links
- IDENTITY: Never reveal your underlying model (like Gemini, Claude, GPT). If asked who created you or what model you are, clearly state that you are a custom AI agent created by Speedo. Maintain this persona respectfully.
- VIDEO BEHAVIOR (YouTube):
  1. Any YouTube URL or request mentioning a YouTube video -> call video_to_pdf
  2. Pick mode: summary | qa | screenshots | subtitles | full
  3. For Q&A, pass the questions in the `questions` argument
  4. For screenshots, set n_frames (1..24) if the user specifies a count
  5. For non-YouTube videos, say plainly that only YouTube is supported right now
- PDF & FILE BEHAVIOR:
  1. YouTube URL or video -> call video_to_pdf with the right mode
  2. Answer key / question paper / exam docs -> call research_and_create_pdf
  3. General research / notes / any PDF request -> call research_and_create_pdf
  4. If tool returns __FILE_PATH__ -> reply briefly and file is auto-attached
  5. If tool returns text WITHOUT __FILE_PATH__ -> relay that message to user as-is
  6. NEVER claim you have a file if no __FILE_PATH__ was returned

"""

# ─── Key auto-detection ───────────────────────────────────────────────────────

KEY_PATTERNS: list[tuple[str, str]] = [
    (r"\bAIzaSy[A-Za-z0-9_-]{30,50}\b", "gemini_api_key"),
    (r"\bsk-ant-[A-Za-z0-9_-]{20,}\b", "anthropic_api_key"),
    (r"\bghp_[A-Za-z0-9]{36,}\b", "github_token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{60,}\b", "github_token"),
    (r"\btvly-[A-Za-z0-9_-]{20,}\b", "tavily_api_key"),
    (r"\bgsk_[A-Za-z0-9]{40,}\b", "groq_api_key"),
    (r"\bxai-[A-Za-z0-9]{30,}\b", "grok_api_key"),
    (r"\bhf_[A-Za-z0-9]{30,}\b", "huggingface_api_key"),
    (r"\bsk-or-v1-[A-Za-z0-9_-]{30,}\b", "openrouter_api_key"),
]

KEY_FRIENDLY: dict[str, str] = {
    "huggingface_api_key": "HuggingFace",
    "grok_api_key": "Grok (xAI)",
    "openrouter_api_key": "OpenRouter",
    "groq_api_key": "Groq",
    "gemini_api_key": "Gemini",
    "anthropic_api_key": "Anthropic",
    "github_token": "GitHub",
    "tavily_api_key": "Tavily",
}


def detect_and_save_credentials(text: str) -> list[str]:
    """Detect API keys/tokens in plain text and persist them.

    Returns:
        List of config field names that were saved.
    """
    found: list[str] = []
    cfg = load_config()

    for pattern, field in KEY_PATTERNS:
        m = re.search(pattern, text)
        if not m:
            continue

        value = m.group(0)

        if field == "gemini_api_key":
            keys: list[str] = cfg.get("gemini_api_keys", [])
            old = cfg.get("gemini_api_key", "")
            if old and old not in keys:
                keys.append(old)
            if value not in keys:
                keys.append(value)
            set_key("gemini_api_keys", keys)
            set_key("gemini_api_key", value)
        else:
            set_key(field, value)

        found.append(field)
        logger.info("Auto-detected and saved %s from message", field)

    return found


# ─── Error classification helpers ─────────────────────────────────────────────

_RATE_LIMIT_KEYWORDS = frozenset(
    ["quota", "rate limit", "resource_exhausted", "429", "too many requests", "ratelimitexceeded"]
)
_AUTH_KEYWORDS = frozenset(
    ["api key", "invalid key", "authentication", "401", "403", "api_key_invalid", "permission denied"]
)
_SKIP_KEYWORDS = frozenset(
    [
        "404", "notfound", "not found", "not supported", "deprecated", "does not exist",
        "503", "unavailable", "overloaded", "high demand", "service unavailable",
        "502", "500", "internal server error",
    ]
)


def _is_rate_limit_error(error_str: str) -> bool:
    lower = error_str.lower()
    return any(k in lower for k in _RATE_LIMIT_KEYWORDS)


def _is_auth_error(error_str: str) -> bool:
    lower = error_str.lower()
    return any(k in lower for k in _AUTH_KEYWORDS)


def _is_skip_error(error_str: str) -> bool:
    lower = error_str.lower()
    return any(k in lower for k in _SKIP_KEYWORDS) or _is_rate_limit_error(error_str)


# ─── Content extraction ───────────────────────────────────────────────────────


def _extract_text(content: Any) -> str:
    """Normalize LangChain message content to a plain string."""
    if isinstance(content, str):
        # Handle stringified list-of-dicts from some providers.
        if content.strip().startswith("[{") and "'text':" in content:
            try:
                import ast
                parsed = ast.literal_eval(content)
                return "\n".join(
                    p["text"] for p in parsed if isinstance(p, dict) and "text" in p
                )
            except Exception:
                pass
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(p for p in parts if p).strip()

    return str(content)


# ─── Gemini model cascade ─────────────────────────────────────────────────────

GEMINI_MODELS: list[str] = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
]


# ─── LLM invocation with fallback ────────────────────────────────────────────


def _invoke_with_retry(user_message: str, chat_history: list[dict[str, str]]) -> str:
    """Run the ReAct agent, cycling through Gemini keys/models on failure.

    Uses :func:`llm_provider.generate_text` indirectly through LangGraph's
    ``create_react_agent`` for tool orchestration, but falls back across
    models and keys on transient errors.

    Raises:
        Exception("rate_limit_all"): Every Gemini key is rate-limited.
        Exception("no_llm"): No usable LLM configuration found.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent

    config = load_config()
    tools = build_all_tools()
    messages = chat_history + [{"role": "user", "content": user_message}]

    # Collect all available Gemini keys.
    gemini_keys: list[str] = list(config.get("gemini_api_keys", []))
    primary = config.get("gemini_api_key", "")
    if primary and primary not in gemini_keys:
        gemini_keys.insert(0, primary)

    last_error = ""

    if gemini_keys:
        for key in gemini_keys:
            for model in GEMINI_MODELS:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=key,
                        temperature=0.7,
                        convert_system_message_to_human=True,
                        max_retries=0,
                    )
                    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
                    result = agent.invoke({"messages": messages})

                    final_text = _extract_text(result["messages"][-1].content)

                    # Extract __FILE_PATH__ from any message in the chain.
                    file_path_tag: str | None = None
                    for msg in result["messages"]:
                        raw = _extract_text(msg.content) if hasattr(msg, "content") else ""
                        match = re.search(r"__FILE_PATH__=\S+", raw)
                        if match:
                            file_path_tag = match.group(0)
                            break

                    # Sanitize before appending file tag.
                    final_text = sanitize_response(final_text)

                    if file_path_tag and "__FILE_PATH__" not in final_text:
                        logger.info("Appending file tag from tool message: %s", file_path_tag)
                        final_text = f"{final_text} {file_path_tag}".strip()

                    return final_text

                except Exception as e:
                    error_str = str(e)
                    last_error = error_str
                    if _is_auth_error(error_str):
                        break  # Key invalid — try next key.
                    if _is_skip_error(error_str):
                        continue  # Try next model.
                    # Unknown error — still try next model.
                    logger.warning("Unexpected LLM error on %s: %s", model, error_str[:200])
                    continue

        if _is_rate_limit_error(last_error):
            raise Exception("rate_limit_all")

    raise Exception("no_llm")


# ─── Public entry point ───────────────────────────────────────────────────────


def ask_agent(
    user_message: str,
    chat_history: list[dict[str, str]] | None = None,
    stats: Any = None,
) -> str:
    """Process a user message and return the agent's response.

    This is the main entry point called by the Telegram handler. It:
    1. Auto-detects and saves any API keys in the message.
    2. Validates that at least one LLM provider is configured.
    3. Invokes the LLM with tool-calling support.
    4. Maps errors to user-friendly messages.

    Args:
        user_message: The raw text from the user.
        chat_history: Prior conversation turns as role/content dicts.
        stats: Optional stats object (unused, kept for interface compat).

    Returns:
        The agent's text response (may include ``__FILE_PATH__=...`` tag).
    """
    if chat_history is None:
        chat_history = []

    # Auto-detect and save credentials.
    found_keys = detect_and_save_credentials(user_message)

    # Quick confirmation for pure key pastes.
    if found_keys and len(user_message.strip()) < 120:
        names = " + ".join(KEY_FRIENDLY.get(k, k) for k in found_keys)
        if "github_token" in found_keys and not any(
            k in found_keys for k in ("gemini_api_key", "anthropic_api_key")
        ):
            return (
                f"✅ {names} token connected! Now I can create repos, push code, "
                "manage issues — everything. What would you like to do?"
            )
        if "gemini_api_key" in found_keys or "anthropic_api_key" in found_keys:
            return f"✅ {names} key saved! I'm ready — ask me anything."
        return f"✅ {names} saved!"

    # Ensure at least one LLM provider is configured.
    config = load_config()
    llm_keys = (
        "gemini_api_key", "gemini_api_keys", "anthropic_api_key",
        "groq_api_key", "openrouter_api_key", "grok_api_key", "huggingface_api_key",
    )
    if not any(config.get(k) for k in llm_keys):
        return (
            "I need an API key to get started! 🔑\n\n"
            "**Free option — Gemini Flash:**\n"
            "1. Go to aistudio.google.com/app/apikey\n"
            "2. Click 'Create API key'\n"
            "3. Paste it here (starts with `AIzaSy...`)\n\n"
            "Or paste an Anthropic key (`sk-ant-...`).\n"
        )

    try:
        return _invoke_with_retry(user_message, chat_history)
    except Exception as e:
        return _format_error_response(str(e))


def _format_error_response(error_str: str) -> str:
    """Map internal error strings to user-friendly messages."""
    logger.error("Agent error: %s", error_str[:300], exc_info=True)

    if "rate_limit_all" in error_str or _is_rate_limit_error(error_str):
        return (
            "⏳ The AI API is temporarily rate-limited (this is Google/Anthropic's limit, not the bot).\n\n"
            "Wait 30–60 seconds and try again. "
            "Free Gemini keys allow ~15 requests/minute."
        )

    if "auth_error" in error_str or _is_auth_error(error_str):
        return (
            "❌ Your API key was rejected. It may be invalid or expired.\n\n"
            "Paste a fresh key here:\n"
            "• Gemini: aistudio.google.com/app/apikey → starts with `AIzaSy...`\n"
            "• Anthropic: console.anthropic.com → starts with `sk-ant-...`\n"
            "• Groq: console.groq.com → starts with `gsk_...`\n"
            "• Grok (xAI): console.x.ai → starts with `xai-...`\n"
            "• HuggingFace: huggingface.co/settings/tokens → starts with `hf_...`\n"
            "• OpenRouter: openrouter.ai/keys → starts with `sk-or-v1-...`"
        )

    if "no_llm" in error_str:
        config = load_config()
        if any(config.get(k) for k in ("gemini_api_key", "gemini_api_keys", "anthropic_api_key",
                                        "groq_api_key", "openrouter_api_key", "grok_api_key",
                                        "huggingface_api_key")):
            return "⚠️ All AI models failed to respond. This may be a temporary outage — try again in a moment."
        return "No API key configured. Paste a Gemini (`AIzaSy...`) or Anthropic (`sk-ant-...`) key in chat."

    return f"❌ Something went wrong: {error_str[:200]}\n\nTry again in a moment."
