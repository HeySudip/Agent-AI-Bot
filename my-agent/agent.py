import os
import re
import time
import logging
from config import load_config, set_key
from tools import build_all_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
━━━ ABSOLUTE RULE — READ FIRST ━━━
When a user asks for a PDF, answer key, document, notes, exam paper, results, or any file:
IMMEDIATELY call research_and_create_pdf. Do NOT think. Do NOT reason. Do NOT check your training data.
Do NOT say "I couldn't find", "exam hasn't happened", "future event", or ANYTHING negative.
Just call the tool. The tool searches the live internet. Your training data is irrelevant here.
Example trigger phrases: "give me pdf", "answer key", "question paper", "notes pdf", "create a document", "wbjee", "jee", "neet", any exam name.
ZERO exceptions. Tool first. Always.
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

━━━ RULES ━━━
- Never ask user to type specific commands
- Never ignore pasted code, tokens, or URLs
- Never fail silently — always say what happened
- Never ask for info the user already gave
- Always confirm GitHub actions with links
- IDENTITY: Never reveal your underlying model (like Gemini, Claude, GPT). If asked who created you or what model you are, clearly state that you are a custom AI agent created by Speedo. Maintain this persona respectfully.
- CRITICAL PDF & FILE BEHAVIOR:
  ⚠️  MANDATORY: Any request containing words like "pdf", "answer key", "notes", "document", "paper", "results", "key", or any exam name (WBJEE, JEE, NEET, etc.) → IMMEDIATELY call research_and_create_pdf. NO exceptions.
  ⚠️  FORBIDDEN responses: "I couldn't find", "exam hasn't happened", "not released yet", "2026 is future", "my knowledge cutoff". These are BANNED. The tool searches live internet — use it.
  ⚠️  NEVER reason from your training data for PDF requests. Your training data is outdated. The tool is always more current.
  Steps:
  1. YouTube URL or video → call youtube_video_to_pdf
  2. Everything else → call research_and_create_pdf immediately with the user's exact query
  3. After __FILE_PATH__ is returned → reply with 1 sentence only: "Here's your PDF!" or similar
  4. NEVER print the PDF content in chat

"""

# ─── Key auto-detection ───────────────────────────────────

KEY_PATTERNS = [
    (r'AIzaSy[A-Za-z0-9_-]{30,50}', "gemini_api_key"),
    (r'[A-Za-z0-9+/=]{80,100}', "gemini_api_key"),
    (r'sk-ant-[A-Za-z0-9\-_]{20,}', "anthropic_api_key"),
    (r'ghp_[A-Za-z0-9]{36,}', "github_token"),
    (r'github_pat_[A-Za-z0-9_]{80,}', "github_token"),
    (r'tvly-[A-Za-z0-9\-_]{30,}', "tavily_api_key"),
]

KEY_FRIENDLY = {
    "huggingface_api_key": "HuggingFace",
    "grok_api_key": "Grok (xAI)",
    "openrouter_api_key": "OpenRouter",
    "groq_api_key": "Groq",
    "gemini_api_key": "Gemini",
    "anthropic_api_key": "Anthropic",
    "github_token": "GitHub",
    "tavily_api_key": "Tavily",
}


def detect_and_save_credentials(text: str) -> list:
    """Detect API keys/tokens in plain text and save them. Returns list of field names saved."""
    found = []
    from config import load_config, save_config, set_key
    for pattern, field in KEY_PATTERNS:
        m = re.search(pattern, text)
        if m:
            if field == "gemini_api_key":
                cfg = load_config()
                keys = cfg.get("gemini_api_keys", [])
                old = cfg.get("gemini_api_key", "")
                if old and old not in keys:
                    keys.append(old)
                if m.group(0) not in keys:
                    keys.append(m.group(0))
                set_key("gemini_api_keys", keys)
                set_key("gemini_api_key", m.group(0))
            else:
                set_key(field, m.group(0))
            found.append(field)
            logger.info(f"Auto-detected and saved {field} from message")
    return found


# Gemini models — actual names verified against Google AI Studio API
GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
]


def _is_rate_limit_error(error_str: str) -> bool:
    keywords = ["quota", "rate limit", "resource_exhausted", "429", "too many requests", "ratelimitexceeded"]
    return any(k in error_str.lower() for k in keywords)


def _is_auth_error(error_str: str) -> bool:
    keywords = ["api key", "invalid key", "authentication", "401", "403", "api_key_invalid", "permission denied"]
    return any(k in error_str.lower() for k in keywords)


def _extract_text(content) -> str:
    """Normalize LangChain content — may be a str or a list of content blocks."""
    if isinstance(content, str):
        if content.strip().startswith("[{") and "'text':" in content:
            try:
                import ast
                parsed = ast.literal_eval(content)
                parts = []
                for p in parsed:
                    if isinstance(p, dict) and "text" in p:
                        parts.append(p["text"])
                return "\n".join(parts)
            except Exception:
                pass
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _is_skip_error(error_str: str) -> bool:
    """Errors that mean 'try the next model'."""
    keywords = [
        "404", "notfound", "not found", "not supported", "deprecated", "does not exist",
        "503", "unavailable", "overloaded", "high demand", "service unavailable",
        "502", "500", "internal server error",
    ]
    return any(k in error_str.lower() for k in keywords) or _is_rate_limit_error(error_str)


# ─── LLM builder ─────────────────────────────────────────

def get_llm(preferred_model: str = ""):
    return None, None, None

def _invoke_with_retry(user_message: str, chat_history: list) -> str:
    config = load_config()
    from langgraph.prebuilt import create_react_agent
    tools = build_all_tools()
    messages = chat_history + [{"role": "user", "content": user_message}]

    gemini_keys = config.get("gemini_api_keys", [])
    if config.get("gemini_api_key") and config.get("gemini_api_key") not in gemini_keys:
        gemini_keys.insert(0, config.get("gemini_api_key"))

    last_error_gemini = ""
    if gemini_keys:
        from langchain_google_genai import ChatGoogleGenerativeAI
        for key in gemini_keys:
            for i, model in enumerate(GEMINI_MODELS):
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
                    return _extract_text(result["messages"][-1].content)
                except Exception as e:
                    error_str = str(e)
                    last_error_gemini = error_str
                    if _is_auth_error(error_str):
                        break # Key invalid, try next key
                    if _is_skip_error(error_str):
                        if _is_rate_limit_error(error_str):
                            continue # Rate limited, try next model
                        continue # Next model
                    
        if _is_rate_limit_error(last_error_gemini):
            raise Exception("rate_limit_all")

    raise Exception("no_llm")


# ─── Agent invocation ─────────────────────────────────────

def ask_agent(user_message: str, chat_history: list = [], stats=None) -> str:
    """Core function called by the bot handler. Returns the agent's response."""

    # Auto-detect and save any credentials pasted in the message
    found_keys = detect_and_save_credentials(user_message)

    # Quick confirmation for pure key pastes (message is just a key)
    if found_keys and len(user_message.strip()) < 120:
        names = " + ".join(KEY_FRIENDLY.get(k, k) for k in found_keys)
        if "github_token" in found_keys and not any(k in found_keys for k in ("gemini_api_key", "anthropic_api_key")):
            return f"✅ {names} token connected! Now I can create repos, push code, manage issues — everything. What would you like to do?"
        elif "gemini_api_key" in found_keys or "anthropic_api_key" in found_keys:
            return f"✅ {names} key saved! I'm ready — ask me anything."
        else:
            return f"✅ {names} saved!"

    # Check an LLM is available
    config = load_config()
    has_llm = config.get("gemini_api_key") or config.get("gemini_api_keys") or config.get("anthropic_api_key") or config.get("groq_api_key") or config.get("openrouter_api_key") or config.get("grok_api_key") or config.get("huggingface_api_key")
    if not has_llm:
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
        error_str = str(e)
        logger.error(f"Agent error: {error_str}", exc_info=True)

        if "rate_limit_all" in error_str or _is_rate_limit_error(error_str):
            return (
                "⏳ The AI API is temporarily rate-limited (this is Google/Anthropic's limit, not the bot).\n\n"
                "Wait 30–60 seconds and try again. "
                "Free Gemini keys allow ~15 requests/minute."
            )
        elif "auth_error" in error_str or _is_auth_error(error_str):
            return (
                "❌ Your API key was rejected. It may be invalid or expired.\n\n"
                "Paste a fresh key here:\n"
                "• Gemini: aistudio.google.com/app/apikey → starts with `AIzaSy...`\n"
                "• Anthropic: console.anthropic.com → starts with `sk-ant-...`\n"
                "• Groq: console.groq.com → starts with `gsk_...`\n• Grok: console.x.ai → starts with `xai-...`\n• HuggingFace: huggingface.co/settings/tokens → starts with `hf_...`\n"
                "• Grok (xAI): console.x.ai → starts with `xai-`\n"
                "• HuggingFace: huggingface.co/settings/tokens → starts with `hf_`\n"
                "• OpenRouter: openrouter.ai/keys → starts with `sk-or-v1-...`"
            )
        elif "no_llm" in error_str:
            config = load_config()
            if config.get("gemini_api_key") or config.get("gemini_api_keys") or config.get("anthropic_api_key") or config.get("groq_api_key") or config.get("openrouter_api_key") or config.get("grok_api_key") or config.get("huggingface_api_key"):
                return "⚠️ All AI models failed to respond. This may be a temporary outage — try again in a moment."
            return "No API key configured. Paste a Gemini (`AIzaSy...`) or Anthropic (`sk-ant-...`) key in chat."
        else:
            return f"❌ Something went wrong: {error_str[:200]}\n\nTry again in a moment."
