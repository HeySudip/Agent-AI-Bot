import sqlite3
import json
import os
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                message_type TEXT DEFAULT 'text'
            );

            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                total_messages INTEGER DEFAULT 0,
                total_tokens_approx INTEGER DEFAULT 0,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                github_actions INTEGER DEFAULT 0,
                web_searches INTEGER DEFAULT 0,
                urls_summarized INTEGER DEFAULT 0,
                preferred_llm TEXT DEFAULT 'gemini'
            );

            CREATE TABLE IF NOT EXISTS user_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                note TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
                ON conversations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_usage_log_user_id
                ON usage_log(user_id);
        """)
    logger.info("Database initialized")


class ConversationStore:

    def __init__(self, user_id: int, max_length: int = 30):
        self.user_id = user_id
        self.max_length = max_length

    def add_message(self, role: str, content: str, message_type: str = "text"):
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (user_id, role, content, timestamp, message_type) VALUES (?, ?, ?, ?, ?)",
                (self.user_id, role, content, time.time(), message_type)
            )
        self._trim()

    def get_history(self, limit: Optional[int] = None) -> list:
        n = limit or self.max_length
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT role, content FROM conversations
                   WHERE user_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (self.user_id, n)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_history_with_timestamps(self, limit: int = 50) -> list:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT role, content, timestamp, message_type FROM conversations
                   WHERE user_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (self.user_id, limit)
            ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "message_type": r["message_type"],
                "time_str": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for r in reversed(rows)
        ]

    def clear(self):
        with get_connection() as conn:
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (self.user_id,))

    def count(self) -> int:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversations WHERE user_id = ?",
                (self.user_id,)
            ).fetchone()
        return row["cnt"] if row else 0

    def _trim(self):
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT id FROM conversations WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT -1 OFFSET ?""",
                (self.user_id, self.max_length * 2)
            ).fetchall()
            if rows:
                ids = [r["id"] for r in rows]
                conn.execute(
                    f"DELETE FROM conversations WHERE id IN ({','.join('?' * len(ids))})",
                    ids
                )

    def export_text(self) -> str:
        history = self.get_history_with_timestamps(limit=200)
        if not history:
            return "No conversation history found."
        lines = [f"=== Conversation Export for user {self.user_id} ===\n"]
        for msg in history:
            prefix = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"[{msg['time_str']}] {prefix}:\n{msg['content']}\n")
        return "\n".join(lines)


class UserStats:

    def __init__(self, user_id: int):
        self.user_id = user_id

    def upsert(self, username: str = "", first_name: str = "", last_name: str = ""):
        now = time.time()
        with get_connection() as conn:
            existing = conn.execute(
                "SELECT user_id FROM user_stats WHERE user_id = ?", (self.user_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE user_stats
                       SET username=?, first_name=?, last_name=?, last_seen=?,
                           total_messages = total_messages + 1
                       WHERE user_id=?""",
                    (username, first_name, last_name, now, self.user_id)
                )
            else:
                conn.execute(
                    """INSERT INTO user_stats
                       (user_id, username, first_name, last_name, total_messages, first_seen, last_seen)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (self.user_id, username, first_name, last_name, now, now)
                )

    def increment(self, field: str, amount: int = 1):
        valid = {"total_messages", "github_actions", "web_searches", "urls_summarized", "total_tokens_approx"}
        if field not in valid:
            return
        with get_connection() as conn:
            conn.execute(
                f"UPDATE user_stats SET {field} = {field} + ? WHERE user_id = ?",
                (amount, self.user_id)
            )

    def get(self) -> Optional[dict]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_stats WHERE user_id = ?", (self.user_id,)
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def log_action(self, action: str, details: str = ""):
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO usage_log (user_id, action, details, timestamp) VALUES (?, ?, ?, ?)",
                (self.user_id, action, details, time.time())
            )

    def get_recent_actions(self, limit: int = 20) -> list:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT action, details, timestamp FROM usage_log
                   WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
                (self.user_id, limit)
            ).fetchall()
        return [
            {
                "action": r["action"],
                "details": r["details"],
                "time": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            }
            for r in rows
        ]

    def add_note(self, note: str):
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO user_notes (user_id, note, created_at) VALUES (?, ?, ?)",
                (self.user_id, note, time.time())
            )

    def get_notes(self) -> list:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT note, created_at FROM user_notes WHERE user_id = ? ORDER BY created_at DESC",
                (self.user_id,)
            ).fetchall()
        return [
            {
                "note": r["note"],
                "time": datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M"),
            }
            for r in rows
        ]


def get_all_users() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM user_stats ORDER BY total_messages DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_global_stats() -> dict:
    with get_connection() as conn:
        total_users = conn.execute("SELECT COUNT(*) as cnt FROM user_stats").fetchone()["cnt"]
        total_messages = conn.execute(
            "SELECT SUM(total_messages) as s FROM user_stats"
        ).fetchone()["s"] or 0
        total_github = conn.execute(
            "SELECT SUM(github_actions) as s FROM user_stats"
        ).fetchone()["s"] or 0
        total_searches = conn.execute(
            "SELECT SUM(web_searches) as s FROM user_stats"
        ).fetchone()["s"] or 0
        total_urls = conn.execute(
            "SELECT SUM(urls_summarized) as s FROM user_stats"
        ).fetchone()["s"] or 0
    return {
        "total_users": total_users,
        "total_messages": total_messages,
        "total_github_actions": total_github,
        "total_web_searches": total_searches,
        "total_urls_summarized": total_urls,
    }
