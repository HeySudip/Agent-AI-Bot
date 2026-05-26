"""Telegram bot command handlers (/start, /help, /status, admin commands, etc.)."""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from typing import Any

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

from config import (
    add_admin,
    add_allowed_user,
    get_llm_status,
    is_admin,
    is_allowed,
    load_config,
    mask_key,
    remove_admin,
    remove_allowed_user,
)
from memory.store import ConversationStore, UserStats, get_all_users, get_global_stats
from utils.formatting import format_stats, format_uptime, split_message
from utils.rate_limiter import global_rate_limiter

__all__ = ["register_command_handlers"]

logger = logging.getLogger(__name__)

BOT_START_TIME: float = time.time()


# ─── Public Commands ───────────────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — greet the user and show setup instructions."""
    user = update.effective_user
    if not user or not update.message:
        return

    stats = UserStats(user.id)
    stats.upsert(
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
    )

    config = load_config()
    has_llm = config.get("gemini_api_key") or config.get("anthropic_api_key")
    has_gh = config.get("github_token")
    name = user.first_name or "there"

    if not has_llm:
        msg = (
            f"👋 Hey {name}! I'm your AI assistant.\n\n"
            "I can chat about *anything*, search the web in real-time, "
            "summarize URLs, and do *everything* on GitHub — "
            "create repos, push code, manage files, issues, PRs, and more.\n\n"
            "To get started, I need an API key:\n"
            "🆓 **Gemini Flash (free):** Get yours at aistudio.google.com\n"
            "   Just paste the key starting with `AIzaSy...`\n\n"
            "Or if you have an Anthropic key (`sk-ant-...`), that works too!\n\n"
            "Just paste the key directly in chat — no commands needed."
        )
    else:
        gh_line = (
            "✅ GitHub connected"
            if has_gh
            else "🐙 Paste your GitHub token (`ghp_...`) to connect"
        )
        msg = (
            f"👋 Hey {name}! Ready to go.\n\n"
            f"{gh_line}\n\n"
            "Just talk to me naturally. I can:\n"
            "💬 Chat about anything\n"
            "🔍 Search the web in real-time\n"
            "🔗 Summarize any URL or article\n"
            "🐙 Full GitHub control\n"
            "🧮 Calculate, convert, encode, and more\n\n"
            "No commands needed — just say what you want!"
        )

    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — list available commands."""
    if not update.message:
        return
    msg = (
        "📚 *Commands*\n\n"
        "/start — Welcome message\n"
        "/status — Check API keys and bot status\n"
        "/clear — Clear conversation history\n"
        "/history — Export your conversation\n"
        "/stats — Your usage statistics\n"
        "/help — This message\n\n"
        "👑 *Admin commands:*\n"
        "/adminadd <user_id> — Add admin\n"
        "/adminremove <user_id> — Remove admin\n"
        "/allowuser <user_id> — Allow a user\n"
        "/blockuser <user_id> — Block a user\n"
        "/allusers — List all users\n"
        "/globalstats — Bot-wide statistics\n"
        "/broadcast <message> — Send message to all users\n"
        "/ratelimit reset <user_id> — Reset rate limit\n\n"
        "💡 *You don't need commands for anything else.*\n"
        "Just talk naturally!"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show configured keys and uptime."""
    if not update.message:
        return
    config = load_config()

    gemini = config.get("gemini_api_key", "")
    anthropic = config.get("anthropic_api_key", "")
    github = config.get("github_token", "")
    tavily = config.get("tavily_api_key", "")

    active_brain = (
        "Gemini Flash 🟢"
        if gemini
        else ("Claude 🟢" if anthropic else "❌ No LLM key set")
    )
    uptime = format_uptime(time.time() - BOT_START_TIME)

    msg = (
        "⚙️ *Bot Status*\n\n"
        f"🧠 Active brain: {active_brain}\n\n"
        f"🔑 Gemini API: `{mask_key(gemini)}`\n"
        f"🔑 Anthropic: `{mask_key(anthropic)}`\n"
        f"🐙 GitHub: `{mask_key(github)}`\n"
        f"🔍 Tavily: `{mask_key(tavily)}`\n"
        f"🌐 DuckDuckGo: ✅ always on\n\n"
        f"⏱ Uptime: `{uptime}`\n"
        f"👥 Public access: `{config.get('public_access', True)}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear — wipe the user's conversation history."""
    if not update.effective_user or not update.message:
        return
    user_id = update.effective_user.id
    store = ConversationStore(user_id)
    count = store.count()
    store.clear()
    await update.message.reply_text(
        f"🧹 Cleared {count} messages from your conversation history."
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history — export conversation as text or file."""
    if not update.effective_user or not update.message:
        return
    user_id = update.effective_user.id
    store = ConversationStore(user_id)
    history = store.get_history_with_timestamps(limit=100)

    if not history:
        await update.message.reply_text("No conversation history yet.")
        return

    lines = [
        "=== Conversation History ===\n",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ]
    for msg in history:
        prefix = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{msg['time_str']}] {prefix}:\n{msg['content']}\n\n{'─' * 40}\n")

    text = "".join(lines)

    if len(text) > 3000:
        file_obj = io.BytesIO(text.encode("utf-8"))
        file_obj.name = f"history_{user_id}.txt"
        await update.message.reply_document(
            document=file_obj,
            filename=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            caption=f"📜 Your last {len(history)} messages",
        )
    else:
        chunks = split_message(f"📜 *Conversation History:*\n\n{text}")
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode="Markdown")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats — show the user's usage statistics."""
    if not update.effective_user or not update.message:
        return
    user_id = update.effective_user.id
    user_stats = UserStats(user_id)
    data = user_stats.get()
    if not data:
        await update.message.reply_text("No stats yet — send me a message first!")
        return
    msg = format_stats(data)
    await update.message.reply_text(msg, parse_mode="Markdown")


# ─── Admin Commands ────────────────────────────────────────────────────────────


def _require_admin(update: Update) -> bool:
    """Return True if the user is an admin; send denial otherwise."""
    return update.effective_user is not None and is_admin(update.effective_user.id)


def _parse_user_id_arg(context: ContextTypes.DEFAULT_TYPE) -> int | None:
    """Extract a user ID from the first command argument, or None."""
    if not context.args:
        return None
    try:
        return int(context.args[0])
    except ValueError:
        return None


async def cmd_admin_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /adminadd <user_id>."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    uid = _parse_user_id_arg(context)
    if uid is None:
        await update.message.reply_text("Usage: /adminadd <user_id>")
        return
    add_admin(uid)
    await update.message.reply_text(f"✅ User {uid} added as admin.")


async def cmd_admin_remove(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /adminremove <user_id>."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    uid = _parse_user_id_arg(context)
    if uid is None:
        await update.message.reply_text("Usage: /adminremove <user_id>")
        return
    remove_admin(uid)
    await update.message.reply_text(f"✅ User {uid} removed from admins.")


async def cmd_allow_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /allowuser <user_id>."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    uid = _parse_user_id_arg(context)
    if uid is None:
        await update.message.reply_text("Usage: /allowuser <user_id>")
        return
    add_allowed_user(uid)
    await update.message.reply_text(f"✅ User {uid} allowed.")


async def cmd_block_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /blockuser <user_id>."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    uid = _parse_user_id_arg(context)
    if uid is None:
        await update.message.reply_text("Usage: /blockuser <user_id>")
        return
    remove_allowed_user(uid)
    await update.message.reply_text(f"✅ User {uid} blocked.")


async def cmd_all_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /allusers — list all registered users."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    users = get_all_users()
    if not users:
        await update.message.reply_text("No users yet.")
        return

    lines = ["👥 *All Users:*\n"]
    for u in users[:30]:
        name = u.get("first_name") or u.get("username") or "Unknown"
        last_seen_ts = u.get("last_seen", 0)
        last_seen_str = datetime.fromtimestamp(last_seen_ts).strftime("%Y-%m-%d")
        lines.append(
            f"• {name} (`{u['user_id']}`) — {u['total_messages']} msgs, "
            f"last seen {last_seen_str}"
        )

    msg = "\n".join(lines)
    for chunk in split_message(msg):
        await update.message.reply_text(chunk, parse_mode="Markdown")


async def cmd_global_stats(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /globalstats — show aggregate bot statistics."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    stats = get_global_stats()
    uptime = format_uptime(time.time() - BOT_START_TIME)
    msg = (
        "📊 *Global Bot Statistics*\n\n"
        f"👥 Total users: `{stats['total_users']}`\n"
        f"💬 Total messages: `{stats['total_messages']}`\n"
        f"🐙 GitHub actions: `{stats['total_github_actions']}`\n"
        f"🔍 Web searches: `{stats['total_web_searches']}`\n"
        f"🔗 URLs summarized: `{stats['total_urls_summarized']}`\n"
        f"⏱ Uptime: `{uptime}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /broadcast <message> — send a message to all users."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /broadcast <message>")
        return

    message_text = " ".join(context.args)
    users = get_all_users()
    sent = 0
    failed = 0
    for user in users:
        try:
            await context.bot.send_message(
                chat_id=user["user_id"],
                text=f"📢 *Broadcast:*\n\n{message_text}",
                parse_mode="Markdown",
            )
            sent += 1
        except Exception:
            failed += 1

    await update.message.reply_text(
        f"📢 Broadcast complete: {sent} sent, {failed} failed."
    )


async def cmd_rate_limit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ratelimit reset <user_id>."""
    if not update.message:
        return
    if not _require_admin(update):
        await update.message.reply_text("❌ Admin only.")
        return
    if (
        context.args
        and len(context.args) >= 2
        and context.args[0] == "reset"
    ):
        try:
            uid = int(context.args[1])
        except ValueError:
            await update.message.reply_text("Invalid user ID.")
            return
        global_rate_limiter.reset(uid)
        await update.message.reply_text(f"✅ Rate limit reset for user {uid}.")
    else:
        await update.message.reply_text("Usage: /ratelimit reset <user_id>")


# ─── Registration ──────────────────────────────────────────────────────────────


def register_command_handlers(app: Any) -> None:
    """Register all command handlers on the Telegram Application."""
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("adminadd", cmd_admin_add))
    app.add_handler(CommandHandler("adminremove", cmd_admin_remove))
    app.add_handler(CommandHandler("allowuser", cmd_allow_user))
    app.add_handler(CommandHandler("blockuser", cmd_block_user))
    app.add_handler(CommandHandler("allusers", cmd_all_users))
    app.add_handler(CommandHandler("globalstats", cmd_global_stats))
    app.add_handler(CommandHandler("broadcast", cmd_broadcast))
    app.add_handler(CommandHandler("ratelimit", cmd_rate_limit))
