import os
import re
import io
import asyncio
import logging
from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters
from telegram.constants import ChatAction
from config import is_allowed
from agent import ask_agent
from memory.store import ConversationStore, UserStats
from utils.formatting import split_message, format_for_telegram
from utils.rate_limiter import global_rate_limiter

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+'
)

async def _keep_typing(bot, chat_id: int, stop_event: asyncio.Event):
    """Keep sending typing action every 4 seconds until stop_event is set."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4.0)
        except asyncio.TimeoutError:
            pass

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    user_message = update.message.text

    if not is_allowed(user_id):
        await update.message.reply_text("⛔ You don't have access to this bot. Contact the admin.")
        return

    allowed, wait_time, reason = global_rate_limiter.is_allowed(user_id)
    if not allowed:
        await update.message.reply_text(f"⏳ {reason}\nPlease wait before sending another message.")
        return

    stats = UserStats(user_id)
    stats.upsert(
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
    )
    stats.log_action("message", user_message[:100])

    store = ConversationStore(user_id)
    history = store.get_history()

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(context.bot, update.effective_chat.id, stop_event)
    )

    try:
        reply = await asyncio.get_event_loop().run_in_executor(
            None, ask_agent, user_message, history, stats
        )

        lower = user_message.lower()
        if any(w in lower for w in ["search", "news", "latest", "find", "look up", "google"]):
            stats.increment("web_searches")
        if URL_PATTERN.search(user_message):
            stats.increment("urls_summarized")
        if any(w in lower for w in ["github", "repo", "branch", "commit", "push", "pull request", "issue", "fork", "gist"]):
            stats.increment("github_actions")

        # Check for file path
        file_path_match = re.search(r"__FILE_PATH__=([\w\./-]+)", reply)
        filepath = None
        if file_path_match:
            filepath = file_path_match.group(1)
            reply = reply.replace(file_path_match.group(0), "").strip()

        store.add_message("user", user_message)
        store.add_message("assistant", reply)

        formatted = format_for_telegram(reply)
        chunks = split_message(formatted)

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                try:
                    await update.message.reply_text(chunk, parse_mode="Markdown")
                except Exception:
                    await update.message.reply_text(chunk)
        
        # Send document if agent generated one
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                await update.message.reply_document(document=f)

    except Exception as e:
        logger.exception(f"Error processing message from user {user_id}")
        error_msg = str(e)
        if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
            await update.message.reply_text(
                "❌ API key error. Please check your key with /status\n"
                "You can paste a new key directly in chat."
            )
        elif "rate limit" in error_msg.lower():
            await update.message.reply_text(
                "⏳ The AI is rate-limited right now. Wait a moment and try again."
            )
        else:
            await update.message.reply_text(
                f"❌ Something went wrong: {error_msg[:200]}\n\nTry again in a moment."
            )
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id

    if not is_allowed(user_id):
        await update.message.reply_text("⛔ Access denied.")
        return

    doc = update.message.document
    caption = update.message.caption or ""

    allowed_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
        ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".xml",
        ".sh", ".bash", ".rs", ".go", ".java", ".cpp", ".c", ".h",
        ".php", ".rb", ".swift", ".kt", ".sql", ".env", ".toml",
    }
    filename = doc.file_name or "file"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed_extensions:
        await update.message.reply_text(
            f"I can read text-based files like .py, .js, .md, .txt, .json, etc.\n"
            f"'{filename}' doesn't look like a text file I can process."
        )
        return

    if doc.file_size > 500_000:
        await update.message.reply_text(
            f"File is too large ({doc.file_size // 1024} KB). Max 500 KB."
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        file = await context.bot.get_file(doc.file_id)
        file_bytes = await file.download_as_bytearray()
        content = file_bytes.decode("utf-8", errors="replace")

        instruction = caption or f"Here is the file '{filename}'. What do you want me to do with it?"
        user_message = f"{instruction}\n\nFile: `{filename}`\n\n```\n{content[:8000]}\n```"
        if len(content) > 8000:
            user_message += f"\n\n_(file truncated — {len(content)} total chars)_"

        stats = UserStats(user_id)
        store = ConversationStore(user_id)
        history = store.get_history()

        reply = await asyncio.get_event_loop().run_in_executor(
            None, ask_agent, user_message, history, stats
        )

        file_path_match = re.search(r"__FILE_PATH__=([\w\./-]+)", reply)
        filepath = None
        if file_path_match:
            filepath = file_path_match.group(1)
            reply = reply.replace(file_path_match.group(0), "").strip()

        store.add_message("user", f"[Uploaded file: {filename}] {caption}")
        store.add_message("assistant", reply)

        chunks = split_message(format_for_telegram(reply))
        for chunk in chunks:
            if chunk.strip():
                try:
                    await update.message.reply_text(chunk, parse_mode="Markdown")
                except Exception:
                    await update.message.reply_text(chunk)
                    
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                await update.message.reply_document(document=f)

    except Exception as e:
        logger.exception("Error handling document")
        await update.message.reply_text(f"❌ Error reading file: {str(e)}")

def register_message_handlers(app):
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
