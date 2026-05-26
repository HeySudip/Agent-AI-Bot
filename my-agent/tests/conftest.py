"""Shared pytest fixtures.

Adds the package root (``my-agent/``) to ``sys.path`` so tests can import
modules directly without packaging the project. This mirrors how the bot
itself is launched (``python bot.py`` from inside ``my-agent/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
