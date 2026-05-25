import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from telegram.ext import Application
from memory.store import init_db
from handlers import register_command_handlers, register_message_handlers

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def build_app() -> Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
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
