"""Telegram bot application entry point."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from telegram.ext import Application

from handlers import register_command_handlers, register_message_handlers
from logging_config import configure_logging, get_logger
from memory.store import init_db
from settings import load_settings

load_dotenv()

settings = load_settings()
configure_logging(settings.log_level, fmt=settings.log_format or None)

logger = get_logger(__name__)


def build_app() -> Application:  # type: ignore[type-arg]
    """Construct and configure the Telegram Application instance."""
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    app: Application = (  # type: ignore[type-arg]
        Application.builder()
        .token(token)
        .concurrent_updates(True)
        .build()
    )

    register_command_handlers(app)
    register_message_handlers(app)
    return app


def main() -> None:
    """Initialize the database and start polling for updates."""
    logger.info("Initializing database...")
    init_db()

    logger.info("Building Telegram application...")
    app = build_app()

    logger.info("🤖 Bot is running. Press Ctrl+C to stop.")
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message", "callback_query"],
    )


if __name__ == "__main__":
    main()
