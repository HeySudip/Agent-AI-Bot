import json
import os
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config/config.json")

DEFAULT_CONFIG = {
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


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **data}
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> bool:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError as e:
        logger.error(f"Failed to save config: {e}")
        return False


def set_key(key: str, value) -> bool:
    config = load_config()
    config[key] = value
    return save_config(config)


def get_key(key: str, default=None):
    return load_config().get(key, default)


def is_admin(user_id: int) -> bool:
    admins = get_key("admin_user_ids", [])
    return user_id in admins


def is_allowed(user_id: int) -> bool:
    if get_key("public_access", True):
        return True
    admins = get_key("admin_user_ids", [])
    allowed = get_key("allowed_user_ids", [])
    return user_id in admins or user_id in allowed


def add_admin(user_id: int) -> bool:
    config = load_config()
    if user_id not in config["admin_user_ids"]:
        config["admin_user_ids"].append(user_id)
        return save_config(config)
    return True


def remove_admin(user_id: int) -> bool:
    config = load_config()
    if user_id in config["admin_user_ids"]:
        config["admin_user_ids"].remove(user_id)
        return save_config(config)
    return True


def add_allowed_user(user_id: int) -> bool:
    config = load_config()
    if user_id not in config["allowed_user_ids"]:
        config["allowed_user_ids"].append(user_id)
        return save_config(config)
    return True


def remove_allowed_user(user_id: int) -> bool:
    config = load_config()
    if user_id in config["allowed_user_ids"]:
        config["allowed_user_ids"].remove(user_id)
        return save_config(config)
    return True


def get_llm_status() -> dict:
    config = load_config()
    return {
        "gemini": bool(config.get("gemini_api_key")),
        "anthropic": bool(config.get("anthropic_api_key")),
        "github": bool(config.get("github_token")),
        "tavily": bool(config.get("tavily_api_key")),
    }


def mask_key(val: str) -> str:
    if not val:
        return "not set"
    if len(val) <= 8:
        return "****"
    return f"{val[:5]}...{val[-4:]}"
