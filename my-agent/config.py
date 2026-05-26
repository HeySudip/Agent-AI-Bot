"""Thread-safe JSON config management for chat-supplied API keys.

The config file is written to disk whenever a key changes so that values
persist across container restarts. A threading lock serializes all
read-modify-write cycles to prevent data races from concurrent Telegram
updates.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

__all__ = [
    "load_config",
    "save_config",
    "set_key",
    "get_key",
    "is_admin",
    "is_allowed",
    "add_admin",
    "remove_admin",
    "add_allowed_user",
    "remove_allowed_user",
    "get_llm_status",
    "mask_key",
]

logger = logging.getLogger(__name__)

CONFIG_FILE: str = os.path.join(os.path.dirname(__file__), "config/config.json")

_lock = threading.Lock()

DEFAULT_CONFIG: dict[str, Any] = {
    "gemini_api_key": "",
    "anthropic_api_key": "",
    "github_token": "",
    "tavily_api_key": "",
    "admin_user_ids": [],
    "allowed_user_ids": [],
    "public_access": True,
    "max_history_length": 30,
    "default_language": "en",
    "response_style": "balanced",
}


def load_config() -> dict[str, Any]:
    """Load config from disk, merging with defaults for missing keys."""
    with _lock:
        return _load_config_unlocked()


def _load_config_unlocked() -> dict[str, Any]:
    """Internal load without acquiring the lock (caller must hold it)."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **data}
        except (json.JSONDecodeError, IOError) as exc:
            logger.error("Failed to load config: %s", exc)
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> bool:
    """Persist *config* to disk. Returns True on success."""
    with _lock:
        return _save_config_unlocked(config)


def _save_config_unlocked(config: dict[str, Any]) -> bool:
    """Internal save without acquiring the lock (caller must hold it)."""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError as exc:
        logger.error("Failed to save config: %s", exc)
        return False


def set_key(key: str, value: Any) -> bool:
    """Set a single config key and persist to disk."""
    with _lock:
        config = _load_config_unlocked()
        config[key] = value
        return _save_config_unlocked(config)


def get_key(key: str, default: Any = None) -> Any:
    """Read a single config key."""
    return load_config().get(key, default)


def is_admin(user_id: int) -> bool:
    """Return True if *user_id* is in the admin list."""
    admins: list[int] = get_key("admin_user_ids", [])
    return user_id in admins


def is_allowed(user_id: int) -> bool:
    """Return True if *user_id* may use the bot."""
    if get_key("public_access", True):
        return True
    admins: list[int] = get_key("admin_user_ids", [])
    allowed: list[int] = get_key("allowed_user_ids", [])
    return user_id in admins or user_id in allowed


def add_admin(user_id: int) -> bool:
    """Add *user_id* to the admin list."""
    with _lock:
        config = _load_config_unlocked()
        if user_id not in config["admin_user_ids"]:
            config["admin_user_ids"].append(user_id)
            return _save_config_unlocked(config)
        return True


def remove_admin(user_id: int) -> bool:
    """Remove *user_id* from the admin list."""
    with _lock:
        config = _load_config_unlocked()
        if user_id in config["admin_user_ids"]:
            config["admin_user_ids"].remove(user_id)
            return _save_config_unlocked(config)
        return True


def add_allowed_user(user_id: int) -> bool:
    """Add *user_id* to the allowed-users list."""
    with _lock:
        config = _load_config_unlocked()
        if user_id not in config["allowed_user_ids"]:
            config["allowed_user_ids"].append(user_id)
            return _save_config_unlocked(config)
        return True


def remove_allowed_user(user_id: int) -> bool:
    """Remove *user_id* from the allowed-users list."""
    with _lock:
        config = _load_config_unlocked()
        if user_id in config["allowed_user_ids"]:
            config["allowed_user_ids"].remove(user_id)
            return _save_config_unlocked(config)
        return True


def get_llm_status() -> dict[str, bool]:
    """Return a dict indicating which provider keys are configured."""
    config = load_config()
    return {
        "gemini": bool(config.get("gemini_api_key")),
        "anthropic": bool(config.get("anthropic_api_key")),
        "github": bool(config.get("github_token")),
        "tavily": bool(config.get("tavily_api_key")),
    }


def mask_key(val: str) -> str:
    """Return a masked representation of an API key for display."""
    if not val:
        return "not set"
    if len(val) <= 8:
        return "****"
    return f"{val[:5]}...{val[-4:]}"
