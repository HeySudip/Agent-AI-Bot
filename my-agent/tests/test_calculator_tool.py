"""Smoke test that the calculator tool now uses safe_eval and rejects unsafe input."""

from __future__ import annotations

import sys

# Provide a minimal stub for ``langchain.tools.tool`` so importing
# ``tools.utilities`` does not require the heavy LangChain runtime in unit
# tests. The decorator returns the wrapped function unchanged.
import types
from collections.abc import Callable

if "langchain" not in sys.modules:
    langchain_pkg = types.ModuleType("langchain")
    tools_module = types.ModuleType("langchain.tools")

    def _identity(fn: Callable[..., object]) -> Callable[..., object]:
        return fn

    tools_module.tool = _identity  # type: ignore[attr-defined]
    langchain_pkg.tools = tools_module  # type: ignore[attr-defined]
    sys.modules["langchain"] = langchain_pkg
    sys.modules["langchain.tools"] = tools_module

# Likewise stub the project's ``config`` module if its disk-backed JSON file
# is not present in the test environment.
if "config" not in sys.modules:
    cfg_module = types.ModuleType("config")
    cfg_module.load_config = lambda: {}  # type: ignore[attr-defined]
    cfg_module.save_config = lambda _cfg: True  # type: ignore[attr-defined]
    cfg_module.set_key = lambda _k, _v: True  # type: ignore[attr-defined]
    cfg_module.get_key = lambda _k, default=None: default  # type: ignore[attr-defined]
    sys.modules["config"] = cfg_module

import importlib.util
import pathlib

_UTILS_PATH = pathlib.Path(__file__).resolve().parent.parent / "tools" / "utilities.py"
_spec = importlib.util.spec_from_file_location("utilities_under_test", _UTILS_PATH)
assert _spec is not None and _spec.loader is not None
_utilities = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utilities)
build_utility_tools = _utilities.build_utility_tools


def _calc():
    tools = build_utility_tools()
    fns = {t.__name__: t for t in tools}
    return fns["calculate"]


def test_calculate_basic_arithmetic() -> None:
    calc = _calc()
    assert calc("2 + 3 * 4") == "14"


def test_calculate_returns_int_for_integer_floats() -> None:
    calc = _calc()
    assert calc("4 / 2") == "2"


def test_calculate_rejects_dunder() -> None:
    calc = _calc()
    out = calc("__import__('os').system('id')")
    assert out.startswith("Error:")


def test_calculate_rejects_attribute_access() -> None:
    calc = _calc()
    out = calc("(1).bit_length()")
    assert out.startswith("Error:")


def test_calculate_handles_division_by_zero() -> None:
    calc = _calc()
    out = calc("1 / 0")
    assert "zero" in out.lower()
