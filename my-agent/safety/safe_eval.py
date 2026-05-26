"""Safe arithmetic expression evaluator.

Replaces the previous use of ``eval()`` in the calculator tool. Evaluates a
numeric expression by walking the Python AST and rejecting any node that
isn't on a strict allow-list. Supports common math operators and a small
set of pure functions from the ``math`` module.

This module accepts NO names, NO attribute access, NO calls to user-defined
callables, NO imports, and NO comprehensions. If the input is anything other
than basic arithmetic over numbers and the allow-listed functions, evaluation
raises :class:`SafeEvalError`.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any, Final

__all__ = ["SafeEvalError", "safe_eval"]


class SafeEvalError(ValueError):
    """Raised when an expression contains disallowed syntax or names."""


_BIN_OPS: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Final[dict[type[ast.unaryop], Callable[[Any], Any]]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "comb": math.comb,
    "perm": math.perm,
}

_ALLOWED_CONSTANTS: Final[dict[str, float]] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_MAX_EXPR_LEN: Final[int] = 256
_MAX_POWER_EXPONENT: Final[int] = 1000


def safe_eval(expression: str) -> int | float:
    """Evaluate a constrained arithmetic expression.

    Args:
        expression: A string such as ``"2 + 3 * sqrt(16)"``.

    Returns:
        The numeric result.

    Raises:
        SafeEvalError: If the expression is malformed, too long, contains
            disallowed nodes, or evaluation overflows.
    """
    if not isinstance(expression, str):
        raise SafeEvalError("Expression must be a string.")
    expression = expression.strip()
    if not expression:
        raise SafeEvalError("Expression is empty.")
    if len(expression) > _MAX_EXPR_LEN:
        raise SafeEvalError(
            f"Expression too long ({len(expression)} > {_MAX_EXPR_LEN} chars)."
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeEvalError(f"Invalid syntax: {exc.msg}") from exc

    try:
        result = _eval_node(tree.body)
    except SafeEvalError:
        raise
    except ZeroDivisionError as exc:
        raise SafeEvalError("Division by zero.") from exc
    except OverflowError as exc:
        raise SafeEvalError("Numeric overflow.") from exc
    except (TypeError, ValueError) as exc:
        raise SafeEvalError(f"Evaluation error: {exc}") from exc

    if not isinstance(result, (int, float)):
        raise SafeEvalError(
            f"Expression must yield a number, got {type(result).__name__}."
        )
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise SafeEvalError("Result is not a finite number.")
    return result


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise SafeEvalError(
            f"Disallowed literal type: {type(node.value).__name__}."
        )

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise SafeEvalError(f"Unknown name: {node.id!r}.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise SafeEvalError(f"Disallowed operator: {op_type.__name__}.")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)):
            if abs(right) > _MAX_POWER_EXPONENT:
                raise SafeEvalError("Exponent too large.")
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise SafeEvalError(f"Disallowed unary op: {op_type.__name__}.")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SafeEvalError("Only direct function calls by name are allowed.")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCTIONS:
            raise SafeEvalError(f"Function not allowed: {fname!r}.")
        if node.keywords:
            raise SafeEvalError("Keyword arguments are not allowed.")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCTIONS[fname](*args)

    raise SafeEvalError(f"Disallowed expression node: {type(node).__name__}.")
