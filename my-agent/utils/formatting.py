import re
from datetime import datetime

MAX_TELEGRAM_LENGTH = 4096
CODE_LANGS = {
    "python", "javascript", "typescript", "js", "ts", "py",
    "java", "c", "cpp", "c++", "go", "rust", "bash", "sh",
    "html", "css", "json", "yaml", "yml", "sql", "ruby",
    "php", "swift", "kotlin", "r", "matlab", "scala",
}


def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"\_*[]()~`>#+-=|{}.!"
    for ch in special:
        text = text.replace(ch, f"\\{ch}")
    return text


def format_for_telegram(text: str) -> str:
    """
    Convert markdown-style text to Telegram-compatible format.
    Handles code blocks, bold, italic, links gracefully.
    """
    if not text:
        return text

    lines = text.split("\n")
    output = []
    in_code_block = False
    code_lang = ""
    code_lines = []

    for line in lines:
        # Detect code block start
        code_start = re.match(r"^```(\w*)\s*$", line)
        if code_start and not in_code_block:
            in_code_block = True
            code_lang = code_start.group(1).lower()
            code_lines = []
            continue

        # Detect code block end
        if line.strip() == "```" and in_code_block:
            in_code_block = False
            block = "\n".join(code_lines)
            lang_label = f" ({code_lang})" if code_lang in CODE_LANGS else ""
            output.append(f"```{lang_label}\n{block}\n```")
            code_lines = []
            code_lang = ""
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Inline code
        line = re.sub(r"`([^`]+)`", r"`\1`", line)

        # Bold: **text** or __text__
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)
        line = re.sub(r"__(.+?)__", r"*\1*", line)

        # Italic: *text* or _text_ (be careful not to double-process)
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", line)

        output.append(line)

    if in_code_block and code_lines:
        output.append(f"```\n{chr(10).join(code_lines)}\n```")

    return "\n".join(output)


def split_message(text: str, max_len: int = MAX_TELEGRAM_LENGTH) -> list:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    in_code = False

    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code

        candidate = current + ("\n" if current else "") + line

        if len(candidate) > max_len:
            if current:
                if in_code:
                    current += "\n```"
                chunks.append(current)
                current = ""
                if in_code:
                    current = "```\n" + line
                else:
                    current = line
            else:
                for i in range(0, len(line), max_len):
                    chunks.append(line[i:i + max_len])
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks or [text[:max_len]]


def format_repo_list(repos: list) -> str:
    if not repos:
        return "No repositories found."
    lines = []
    for r in repos:
        vis = "🔒" if r.get("private") else "🌍"
        star = r.get("stargazers_count", 0)
        desc = r.get("description") or ""
        desc_short = (desc[:60] + "…") if len(desc) > 60 else desc
        lines.append(f"{vis} [{r['full_name']}]({r['html_url']}) ⭐{star}")
        if desc_short:
            lines.append(f"   _{desc_short}_")
    return "\n".join(lines)


def format_stats(stats: dict) -> str:
    if not stats:
        return "No stats available yet."
    first = datetime.fromtimestamp(stats.get("first_seen", 0)).strftime("%Y-%m-%d")
    last = datetime.fromtimestamp(stats.get("last_seen", 0)).strftime("%Y-%m-%d %H:%M")
    name = stats.get("first_name", "") or stats.get("username", "") or "User"
    return (
        f"📊 *Stats for {name}*\n\n"
        f"💬 Messages sent: `{stats.get('total_messages', 0)}`\n"
        f"🐙 GitHub actions: `{stats.get('github_actions', 0)}`\n"
        f"🔍 Web searches: `{stats.get('web_searches', 0)}`\n"
        f"🔗 URLs summarized: `{stats.get('urls_summarized', 0)}`\n"
        f"📅 First seen: `{first}`\n"
        f"🕐 Last seen: `{last}`"
    )


def format_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def detect_language(code: str) -> str:
    patterns = {
        "python": [r"def ", r"import ", r"print\(", r"class ", r"elif "],
        "javascript": [r"const ", r"let ", r"var ", r"function ", r"=>\s*{", r"console\.log"],
        "typescript": [r": string", r": number", r": boolean", r"interface ", r"type "],
        "java": [r"public class", r"System\.out", r"void main"],
        "rust": [r"fn main", r"let mut", r"println!", r"use std"],
        "go": [r"func main", r"fmt\.Print", r"package main"],
        "html": [r"<html", r"<div", r"<body", r"<!DOCTYPE"],
        "css": [r"\{[\s\S]*:[\s\S]*;", r"@media", r"\.[\w-]+\s*\{"],
        "bash": [r"#!/bin", r"\$\(", r"echo ", r"if \["],
        "sql": [r"SELECT ", r"INSERT ", r"UPDATE ", r"CREATE TABLE"],
        "json": [r'^\s*\{', r'^\s*\[', r'":\s*["{0-9\[{]'],
    }
    code_lower = code.lower()
    for lang, pats in patterns.items():
        matches = sum(1 for p in pats if re.search(p, code, re.IGNORECASE))
        if matches >= 2:
            return lang
    return "text"
