import math
import re
import json
import time
import logging
from datetime import datetime, timezone
from langchain.tools import tool
from config import load_config

logger = logging.getLogger(__name__)


def build_utility_tools() -> list:

    @tool
    def calculate(expression: str) -> str:
        """
        Evaluate a mathematical expression safely.
        Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, abs, round, etc.
        Examples: "2 ** 10", "sqrt(144)", "sin(3.14159/2)", "log(100, 10)"
        """
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "log": math.log, "log2": math.log2, "log10": math.log10,
            "exp": math.exp, "pow": math.pow, "factorial": math.factorial,
            "pi": math.pi, "e": math.e, "inf": math.inf,
            "degrees": math.degrees, "radians": math.radians,
            "gcd": math.gcd, "lcm": getattr(math, "lcm", None),
            "hypot": math.hypot, "comb": math.comb, "perm": math.perm,
        }
        allowed_names = {k: v for k, v in allowed_names.items() if v is not None}

        # Sanitize
        clean = re.sub(r'[^0-9+\-*/().,%\s\w]', '', expression)
        try:
            result = eval(clean, {"__builtins__": {}}, allowed_names)
            if isinstance(result, float):
                if result == int(result):
                    return str(int(result))
                return f"{result:.10g}"
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except OverflowError:
            return "Error: Result too large to compute"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    @tool
    def get_current_datetime(timezone_name: str = "UTC") -> str:
        """
        Get the current date and time.
        timezone_name: e.g., 'UTC', 'US/Eastern', 'Asia/Kolkata', 'Europe/London'
        Returns formatted date, time, day of week, and Unix timestamp.
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
        """
        Analyze or format code.
        action options: 'analyze' (detect language, count lines, complexity),
                        'count_lines', 'detect_language'
        """
        if action == "detect_language":
            from utils.formatting import detect_language
            lang = detect_language(code)
            return f"Detected language: **{lang}**"

        lines = code.split("\n")
        non_empty = [l for l in lines if l.strip()]
        comment_lines = [l for l in non_empty if l.strip().startswith(("#", "//", "/*", "*", "<!--"))]
        function_count = len(re.findall(r"\bdef \w+|\bfunction \w+|\bfunc \w+|\bfn \w+", code))
        class_count = len(re.findall(r"\bclass \w+", code))
        import_count = len(re.findall(r"^\s*(import |from |require|use )", code, re.MULTILINE))

        from utils.formatting import detect_language
        lang = detect_language(code)

        return (
            f"**Code Analysis:**\n"
            f"• Language: `{lang}`\n"
            f"• Total lines: `{len(lines)}`\n"
            f"• Non-empty lines: `{len(non_empty)}`\n"
            f"• Comment lines: `{len(comment_lines)}`\n"
            f"• Functions/methods: `{function_count}`\n"
            f"• Classes: `{class_count}`\n"
            f"• Imports: `{import_count}`\n"
            f"• Estimated complexity: `{'simple' if function_count < 5 else 'moderate' if function_count < 20 else 'complex'}`"
        )

    @tool
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert between common units.
        Supports: length (m, km, ft, in, mi, cm, mm),
                  weight (kg, g, lb, oz),
                  temperature (c, f, k),
                  data (b, kb, mb, gb, tb),
                  time (s, ms, min, hr, day)
        """
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()

        # Temperature (special case)
        temp_units = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}
        if from_unit in temp_units or to_unit in temp_units:
            aliases = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}
            f = aliases.get(from_unit, from_unit)
            t = aliases.get(to_unit, to_unit)
            if f == "c" and t == "f":
                result = value * 9 / 5 + 32
            elif f == "f" and t == "c":
                result = (value - 32) * 5 / 9
            elif f == "c" and t == "k":
                result = value + 273.15
            elif f == "k" and t == "c":
                result = value - 273.15
            elif f == "f" and t == "k":
                result = (value - 32) * 5 / 9 + 273.15
            elif f == "k" and t == "f":
                result = (value - 273.15) * 9 / 5 + 32
            else:
                result = value
            return f"{value}°{f.upper()} = {result:.4g}°{t.upper()}"

        # Conversion tables (to base unit)
        tables = {
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
        if from_unit not in tables or to_unit not in tables:
            return f"Unknown unit '{from_unit}' or '{to_unit}'. Supported: {', '.join(sorted(tables.keys()))}"
        base = value * tables[from_unit]
        result = base / tables[to_unit]
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    @tool
    def encode_decode(text: str, operation: str) -> str:
        """
        Encode or decode text in various formats.
        operation: 'base64_encode', 'base64_decode', 'url_encode', 'url_decode',
                   'hex_encode', 'hex_decode', 'json_pretty', 'json_minify', 'count_chars'
        """
        import base64
        import urllib.parse

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
                ops = ["base64_encode", "base64_decode", "url_encode", "url_decode",
                       "hex_encode", "hex_decode", "json_pretty", "json_minify", "count_chars"]
                return f"Unknown operation. Valid: {', '.join(ops)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def generate_text(text_type: str, params: str = "") -> str:
        """
        Generate random useful text content.
        text_type: 'uuid', 'password', 'lorem', 'random_number', 'color'
        params: optional parameters (e.g., length for password, count for lorem words)
        """
        import uuid
        import random
        import string

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
            except Exception:
                return str(random.randint(1, 100))

        elif text_type == "color":
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
            return f"HEX: `{hex_color}` | RGB: `rgb({r}, {g}, {b})`"

        else:
            return "Unknown type. Valid: uuid, password, lorem, random_number, color"

    @tool
    def format_json(json_text: str) -> str:
        """
        Parse and pretty-format JSON text.
        Also validates JSON and reports any errors.
        """
        try:
            parsed = json.loads(json_text)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            key_count = len(parsed) if isinstance(parsed, dict) else len(parsed) if isinstance(parsed, list) else "N/A"
            type_name = type(parsed).__name__
            return (
                f"✅ Valid JSON ({type_name}, {key_count} items)\n\n"
                f"```json\n{pretty}\n```"
            )
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON: {str(e)}"

    @tool
    def compare_texts(text1: str, text2: str) -> str:
        """
        Compare two texts and show differences.
        Returns a diff summary showing added, removed, and changed lines.
        """
        import difflib
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
