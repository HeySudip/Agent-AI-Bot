import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from telegram.ext import Application
from memory.store import init_db
from handlers import register_command_handlers, register_message_handlers
from logging_config import configure_logging, get_logger
from settings import load_settings

load_dotenv()

settings = load_settings()
configure_logging(settings.log_level, fmt=settings.log_format or None)

logger = get_logger(__name__)


def build_app() -> Application:
    token = settings.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    app = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)
        .build()
    )

    register_command_handlers(app)
    register_message_handlers(app)

    return app


def main():
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
