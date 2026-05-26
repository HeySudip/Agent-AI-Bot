"""General-purpose utility tools.

Provides a safe calculator (AST-walking evaluator), date/time lookup,
code analysis, unit conversion, text encoding/decoding, random text
generation, JSON formatting, and text comparison.
"""

from __future__ import annotations

import base64
import difflib
import json
import logging
import random
import re
import string
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain.tools import tool

from safety.safe_eval import SafeEvalError, safe_eval

logger = logging.getLogger(__name__)

__all__ = ["build_utility_tools"]


# ─── Public builder ───────────────────────────────────────────────────────────


def build_utility_tools() -> list:
    """Build and return the list of general-purpose LangChain tools."""

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely.

        Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, abs, round, etc.
        Examples: "2 ** 10", "sqrt(144)", "sin(3.14159/2)", "log(100, 10)"

        Uses an AST-walking evaluator — no eval(). Only numeric literals,
        named constants (pi, e, tau, inf), arithmetic operators, and an
        allow-list of math functions are permitted.
        """
        try:
            result = safe_eval(expression)
        except SafeEvalError as exc:
            return f"Error: {exc}"

        if isinstance(result, float):
            return str(int(result)) if result.is_integer() else f"{result:.10g}"
        return str(result)

    @tool
    def get_current_datetime(timezone_name: str = "UTC") -> str:
        """Get the current date and time.

        Args:
            timezone_name: IANA timezone (e.g. 'UTC', 'US/Eastern', 'Asia/Kolkata').

        Returns:
            Formatted date, time, day of week, timezone, Unix timestamp, and ISO 8601.
        """
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(timezone_name)
            now = datetime.now(tz)
        except Exception:
            now = datetime.now(timezone.utc)
            timezone_name = "UTC"

        return (
            f"📅 **Date:** {now.strftime('%B %d, %Y')}\n"
            f"🕐 **Time:** {now.strftime('%H:%M:%S')}\n"
            f"📆 **Day:** {now.strftime('%A')}\n"
            f"🌍 **Timezone:** {timezone_name}\n"
            f"📟 **Unix Timestamp:** {int(now.timestamp())}\n"
            f"📝 **ISO 8601:** {now.isoformat()}"
        )

    @tool
    def format_and_analyze_code(code: str, action: str = "analyze") -> str:
        """Analyze or format code.

        Args:
            code: Source code text to analyze.
            action: One of 'analyze' (detect language, count lines, complexity),
                    'count_lines', or 'detect_language'.
        """
        from utils.formatting import detect_language

        if action == "detect_language":
            return f"Detected language: **{detect_language(code)}**"

        lines = code.split("\n")
        non_empty = [l for l in lines if l.strip()]
        comment_lines = [
            l for l in non_empty if l.strip().startswith(("#", "//", "/*", "*", "<!--"))
        ]
        function_count = len(re.findall(r"\bdef \w+|\bfunction \w+|\bfunc \w+|\bfn \w+", code))
        class_count = len(re.findall(r"\bclass \w+", code))
        import_count = len(re.findall(r"^\s*(import |from |require|use )", code, re.MULTILINE))

        if function_count < 5:
            complexity = "simple"
        elif function_count < 20:
            complexity = "moderate"
        else:
            complexity = "complex"

        return (
            f"**Code Analysis:**\n"
            f"• Language: `{detect_language(code)}`\n"
            f"• Total lines: `{len(lines)}`\n"
            f"• Non-empty lines: `{len(non_empty)}`\n"
            f"• Comment lines: `{len(comment_lines)}`\n"
            f"• Functions/methods: `{function_count}`\n"
            f"• Classes: `{class_count}`\n"
            f"• Imports: `{import_count}`\n"
            f"• Estimated complexity: `{complexity}`"
        )

    @tool
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert between common units.

        Supports: length (m, km, ft, in, mi, cm, mm, yd),
                  weight (kg, g, mg, lb, oz, ton),
                  temperature (c, f, k),
                  data (b, kb, mb, gb, tb),
                  time (s, ms, min, hr, day, week)
        """
        from_u = from_unit.lower().strip()
        to_u = to_unit.lower().strip()

        # Temperature requires special formulas
        temp_aliases: dict[str, str] = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}
        temp_units = {"c", "f", "k", *temp_aliases}

        if from_u in temp_units or to_u in temp_units:
            f = temp_aliases.get(from_u, from_u)
            t = temp_aliases.get(to_u, to_u)
            result = _convert_temperature(value, f, t)
            return f"{value}°{f.upper()} = {result:.4g}°{t.upper()}"

        # Linear conversion via base-unit tables
        tables: dict[str, float] = {
            # Length → meters
            "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001,
            "ft": 0.3048, "in": 0.0254, "mi": 1609.344, "yd": 0.9144,
            # Weight → grams
            "g": 1, "kg": 1000, "mg": 0.001, "lb": 453.592, "oz": 28.3495, "ton": 1e6,
            # Data → bytes
            "b": 1, "kb": 1024, "mb": 1048576, "gb": 1073741824, "tb": 1099511627776,
            # Time → seconds
            "s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400, "week": 604800,
        }

        if from_u not in tables or to_u not in tables:
            supported = ", ".join(sorted(tables))
            return f"Unknown unit '{from_u}' or '{to_u}'. Supported: {supported}"

        result = value * tables[from_u] / tables[to_u]
        return f"{value} {from_u} = {result:.6g} {to_u}"

    @tool
    def encode_decode(text: str, operation: str) -> str:
        """Encode or decode text in various formats.

        Args:
            text: Input text to process.
            operation: One of 'base64_encode', 'base64_decode', 'url_encode',
                       'url_decode', 'hex_encode', 'hex_decode', 'json_pretty',
                       'json_minify', 'count_chars'.
        """
        _VALID_OPS = [
            "base64_encode", "base64_decode", "url_encode", "url_decode",
            "hex_encode", "hex_decode", "json_pretty", "json_minify", "count_chars",
        ]
        try:
            if operation == "base64_encode":
                return base64.b64encode(text.encode()).decode()
            elif operation == "base64_decode":
                return base64.b64decode(text.encode()).decode()
            elif operation == "url_encode":
                return urllib.parse.quote(text)
            elif operation == "url_decode":
                return urllib.parse.unquote(text)
            elif operation == "hex_encode":
                return text.encode().hex()
            elif operation == "hex_decode":
                return bytes.fromhex(text).decode()
            elif operation == "json_pretty":
                return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            elif operation == "json_minify":
                return json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
            elif operation == "count_chars":
                words = len(text.split())
                chars = len(text)
                chars_no_space = len(text.replace(" ", ""))
                lines = len(text.splitlines())
                return (
                    f"Characters (with spaces): {chars}\n"
                    f"Characters (no spaces): {chars_no_space}\n"
                    f"Words: {words}\n"
                    f"Lines: {lines}"
                )
            else:
                return f"Unknown operation. Valid: {', '.join(_VALID_OPS)}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def generate_text(text_type: str, params: str = "") -> str:
        """Generate random useful text content.

        Args:
            text_type: One of 'uuid', 'password', 'lorem', 'random_number', 'color'.
            params: Optional parameters — length for password, word count for lorem,
                    'min-max' range for random_number.
        """
        if text_type == "uuid":
            return str(uuid.uuid4())

        elif text_type == "password":
            length = int(params) if params.isdigit() else 16
            chars = string.ascii_letters + string.digits + "!@#$%^&*()"
            password = "".join(random.choices(chars, k=length))
            return f"Generated password ({length} chars): `{password}`"

        elif text_type == "lorem":
            words = [
                "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
                "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
                "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua",
                "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation",
                "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
            ]
            count = int(params) if params.isdigit() else 30
            return " ".join(random.choices(words, k=count)).capitalize() + "."

        elif text_type == "random_number":
            try:
                parts = params.split("-") if params else []
                low = int(parts[0]) if len(parts) > 0 else 1
                high = int(parts[1]) if len(parts) > 1 else 100
                return str(random.randint(low, high))
            except (ValueError, IndexError):
                return str(random.randint(1, 100))

        elif text_type == "color":
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
            return f"HEX: `{hex_color}` | RGB: `rgb({r}, {g}, {b})`"

        else:
            return "Unknown type. Valid: uuid, password, lorem, random_number, color"

    @tool
    def format_json(json_text: str) -> str:
        """Parse and pretty-format JSON text. Also validates and reports errors."""
        try:
            parsed = json.loads(json_text)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            if isinstance(parsed, (dict, list)):
                item_count = len(parsed)
            else:
                item_count = "N/A"
            type_name = type(parsed).__name__
            return (
                f"✅ Valid JSON ({type_name}, {item_count} items)\n\n"
                f"```json\n{pretty}\n```"
            )
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON: {e}"

    @tool
    def compare_texts(text1: str, text2: str) -> str:
        """Compare two texts and show differences.

        Returns a unified diff summary showing added and removed lines.
        """
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)
        diff = list(difflib.unified_diff(lines1, lines2, fromfile="text1", tofile="text2", lineterm=""))

        if not diff:
            return "✅ The two texts are identical."

        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

        output = f"📊 **Diff Summary:** +{added} added, -{removed} removed\n\n```\n"
        output += "\n".join(diff[:80])
        if len(diff) > 80:
            output += f"\n... ({len(diff) - 80} more lines)"
        output += "\n```"
        return output

    return [
        calculate,
        get_current_datetime,
        format_and_analyze_code,
        convert_units,
        encode_decode,
        generate_text,
        format_json,
        compare_texts,
    ]


# ─── Private helpers ──────────────────────────────────────────────────────────


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between Celsius (c), Fahrenheit (f), and Kelvin (k)."""
    conversions: dict[tuple[str, str], Any] = {
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("c", "k"): lambda v: v + 273.15,
        ("k", "c"): lambda v: v - 273.15,
        ("f", "k"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("k", "f"): lambda v: (v - 273.15) * 9 / 5 + 32,
    }
    fn = conversions.get((from_unit, to_unit))
    return fn(value) if fn else value
