"""Telegram message handlers for text and document messages."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes, MessageHandler, filters

from agent import ask_agent
from config import is_allowed
from memory.store import ConversationStore, UserStats
from utils.formatting import format_for_telegram, split_message
from utils.rate_limiter import global_rate_limiter

__all__ = ["register_message_handlers"]

logger = logging.getLogger(__name__)

_URL_PATTERN: re.Pattern[str] = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+'
)

_ALLOWED_FILE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".xml",
    ".sh", ".bash", ".rs", ".go", ".java", ".cpp", ".c", ".h",
    ".php", ".rb", ".swift", ".kt", ".sql", ".env", ".toml",
})

_MAX_FILE_SIZE: int = 500_000  # 500 KB


async def _keep_typing(bot: Any, chat_id: int, stop_event: asyncio.Event) -> None:
    """Send typing indicator every 4 seconds until *stop_event* is set."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4.0)
        except asyncio.TimeoutError:
            pass


def _extract_file_path(reply: str) -> tuple[str, str | None]:
    """Extract ``__FILE_PATH__=...`` tag from reply text.

    Returns:
        A tuple of (cleaned_reply, filepath_or_none).
    """
    match = re.search(r"__FILE_PATH__=([^\s]+)", reply)
    if match:
        filepath = match.group(1)
        cleaned = reply.replace(match.group(0), "").strip()
        return cleaned, filepath
    return reply, None


async def _send_reply_chunks(
    update: Update, text: str, filepath: str | None
) -> None:
    """Format and send reply text in chunks, then attach file if present."""
    if not update.message:
        return

    formatted = format_for_telegram(text)
    chunks = split_message(formatted)

    for chunk in chunks:
        if chunk.strip():
            try:
                await update.message.reply_text(chunk, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(chunk)

    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            await update.message.reply_document(document=f)
    elif filepath:
        logger.error("File not found on disk: %s", filepath)


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Process an incoming text message through the agent."""
    user = update.effective_user
    if not user or not update.message or not update.message.text:
        return

    user_id = user.id
    user_message = update.message.text

    if not is_allowed(user_id):
        await update.message.reply_text(
            "⛔ You don't have access to this bot. Contact the admin."
        )
        return

    allowed, wait_time, reason = global_rate_limiter.is_allowed(user_id)
    if not allowed:
        await update.message.reply_text(
            f"⏳ {reason}\nPlease wait before sending another message."
        )
        return

    # Track user info
    stats = UserStats(user_id)
    stats.upsert(
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
    )
    stats.log_action("message", user_message[:100])

    store = ConversationStore(user_id)
    history = store.get_history()

    # Show typing indicator while processing
    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(context.bot, update.effective_chat.id, stop_event)  # type: ignore[union-attr]
    )

    try:
        reply = await asyncio.get_event_loop().run_in_executor(
            None, ask_agent, user_message, history, stats
        )

        # Heuristic stat tracking
        lower = user_message.lower()
        if any(w in lower for w in ("search", "news", "latest", "find", "look up", "google")):
            stats.increment("web_searches")
        if _URL_PATTERN.search(user_message):
            stats.increment("urls_summarized")
        if any(
            w in lower
            for w in (
                "github", "repo", "branch", "commit", "push",
                "pull request", "issue", "fork", "gist",
            )
        ):
            stats.increment("github_actions")

        reply, filepath = _extract_file_path(reply)

        store.add_message("user", user_message)
        store.add_message("assistant", reply)

        await _send_reply_chunks(update, reply, filepath)

    except Exception as exc:
        logger.exception("Error processing message from user %d", user_id)
        error_msg = str(exc).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            await update.message.reply_text(
                "❌ API key error. Please check your key with /status\n"
                "You can paste a new key directly in chat."
            )
        elif "rate limit" in error_msg:
            await update.message.reply_text(
                "⏳ The AI is rate-limited right now. Wait a moment and try again."
            )
        else:
            await update.message.reply_text(
                f"❌ Something went wrong: {str(exc)[:200]}\n\nTry again in a moment."
            )
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def handle_document(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Process an uploaded document through the agent."""
    user = update.effective_user
    if not user or not update.message:
        return

    user_id = user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ Access denied.")
        return

    doc = update.message.document
    if not doc:
        return

    caption = update.message.caption or ""
    filename = doc.file_name or "file"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in _ALLOWED_FILE_EXTENSIONS:
        await update.message.reply_text(
            f"I can read text-based files like .py, .js, .md, .txt, .json, etc.\n"
            f"'{filename}' doesn't look like a text file I can process."
        )
        return

    if doc.file_size and doc.file_size > _MAX_FILE_SIZE:
        await update.message.reply_text(
            f"File is too large ({doc.file_size // 1024} KB). Max 500 KB."
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,  # type: ignore[union-attr]
        action=ChatAction.TYPING,
    )

    try:
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()
        content = file_bytes.decode("utf-8", errors="replace")

        instruction = (
            caption or f"Here is the file '{filename}'. What do you want me to do with it?"
        )
        user_message = f"{instruction}\n\nFile: `{filename}`\n\n```\n{content[:8000]}\n```"
        if len(content) > 8000:
            user_message += f"\n\n_(file truncated — {len(content)} total chars)_"

        stats = UserStats(user_id)
        store = ConversationStore(user_id)
        history = store.get_history()

        reply = await asyncio.get_event_loop().run_in_executor(
            None, ask_agent, user_message, history, stats
        )

        reply, filepath = _extract_file_path(reply)

        store.add_message("user", f"[Uploaded file: {filename}] {caption}")
        store.add_message("assistant", reply)

        await _send_reply_chunks(update, reply, filepath)

    except Exception as exc:
        logger.exception("Error handling document from user %d", user_id)
        await update.message.reply_text(f"❌ Error reading file: {exc}")


def register_message_handlers(app: Any) -> None:
    """Register text and document message handlers on the Telegram Application."""
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
