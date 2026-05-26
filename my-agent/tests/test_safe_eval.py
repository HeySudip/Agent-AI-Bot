"""Tests for the safe arithmetic evaluator."""

from __future__ import annotations

import math

import pytest

from safety.safe_eval import SafeEvalError, safe_eval


class TestArithmetic:
    def test_addition(self) -> None:
        assert safe_eval("1 + 2") == 3

    def test_unary_minus(self) -> None:
        assert safe_eval("-5 + 10") == 5

    def test_power(self) -> None:
        assert safe_eval("2 ** 10") == 1024

    def test_floor_div_and_mod(self) -> None:
        assert safe_eval("17 // 5") == 3
        assert safe_eval("17 % 5") == 2

    def test_precedence(self) -> None:
        assert safe_eval("2 + 3 * 4") == 14
        assert safe_eval("(2 + 3) * 4") == 20

    def test_floats(self) -> None:
        assert safe_eval("0.1 + 0.2") == pytest.approx(0.3)


class TestFunctions:
    def test_sqrt(self) -> None:
        assert safe_eval("sqrt(144)") == 12

    def test_log_with_base(self) -> None:
        assert safe_eval("log(100, 10)") == pytest.approx(2.0)

    def test_trig(self) -> None:
        assert safe_eval("sin(0)") == 0
        assert safe_eval("cos(0)") == 1

    def test_constant_pi(self) -> None:
        assert safe_eval("pi") == pytest.approx(math.pi)


class TestRejections:
    def test_empty_string(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("")

    def test_non_string(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval(123)  # type: ignore[arg-type]

    def test_oversized_input(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("1+" * 200 + "1")

    def test_division_by_zero(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("1 / 0")

    def test_unknown_name(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("foo + 1")

    def test_attribute_access_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("(1).bit_length()")

    def test_dunder_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("__import__('os')")

    def test_comprehension_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("[i for i in range(10)]")

    def test_lambda_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("(lambda: 1)()")

    def test_string_literal_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("'abc' + 'def'")

    def test_huge_exponent_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("2 ** 100000")

    def test_keyword_args_blocked(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("round(1.5, ndigits=0)")

    def test_disallowed_function(self) -> None:
        with pytest.raises(SafeEvalError):
            safe_eval("print(1)")
