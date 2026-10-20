# pyright: basic
"""
Utilities for parsing indexing expressions in DataBlock and Ensemble.

Numeric axes understand simple comparisons (`x>2`, `x<=5`, `x==3`, `x!=3`) and
compound ones (`-3<=x<3`, `3>x>1`). Numbers may use scientific notation (`1e-3`) and
omit the leading zero (`.5`). The name of the axis may be left out of a simple
comparison (`>2`). Categorical axes understand `category==dog` and `category!=dog`,
where values with spaces or special characters are quoted (`name=='new york'`).
"""

import operator
import re
from typing import Literal, NamedTuple, TypeAlias, cast

import numpy as np

from .Types import Axis1D

OperatorStr: TypeAlias = Literal["<"] | Literal["<="] | Literal[">"] | Literal[">="]

NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
CATEGORY_VALUE = r"""'[^']*'|"[^"]*"|[^\s'"=!<>]+"""
# What may stand where an axis name is expected when we only want to report a mismatch
_ANY_NAME = r"[^\s<>=!]"

_COMPARISONS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_MIN_QUOTED_LENGTH = 2  # the two quote characters
_ASCENDING = ("<", "<=")


def compound_to_mask(
    axis_points: Axis1D,
    val1: float | int,
    val2: float | int,
    op1: OperatorStr,
    op2: OperatorStr,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Mask of `val1 op1 axis_points op2 val2`, for example `val1 < points <= val2`."""
    points = np.asarray(axis_points)
    return _COMPARISONS[op1](val1, points) & _COMPARISONS[op2](points, val2)


def _mismatch(axis_name: str, expr: str, kind: str, variable: str | None) -> ValueError:
    if variable is not None:
        return ValueError(
            f"Variable '{variable}' in expression doesn't match axis '{axis_name}'"
        )
    return ValueError(f"Invalid {kind} expression: '{expr}' for axis '{axis_name}'")


class _Simple(NamedTuple):
    """`axis op value`, numbers as written in the expression"""

    op: str
    value: str


class _Compound(NamedTuple):
    """`value1 op1 axis op2 value2`, numbers as written in the expression"""

    value1: str
    op1: OperatorStr
    op2: OperatorStr
    value2: str


def _parse_inequality(axis_name: str, expr: str) -> _Simple | _Compound:
    """Parses a numeric comparison; raises ValueError for anything else."""
    text = expr.strip()
    name = re.escape(axis_name)

    compound = re.fullmatch(
        rf"({NUMBER})\s*(<=|>=|<|>)\s*{name}\s*(<=|>=|<|>)\s*({NUMBER})", text
    )
    if compound:
        val1, op1, op2, val2 = compound.groups()
        if (op1 in _ASCENDING) != (op2 in _ASCENDING):
            raise ValueError(
                f"Compound expression '{expr}' for axis '{axis_name}' should use "
                "'<' / '<=' on both sides or '>' / '>=' on both sides"
            )
        return _Compound(val1, cast("OperatorStr", op1), cast("OperatorStr", op2), val2)

    simple = re.fullmatch(rf"(?:{name})?\s*(<=|>=|==|!=|<|>)\s*({NUMBER})", text)
    if simple:
        op, val = simple.groups()
        return _Simple(op, val)

    other = re.fullmatch(rf"({_ANY_NAME}*)\s*(?:<=|>=|==|!=|<|>)\s*{NUMBER}", text)
    if other and other.group(1):
        raise _mismatch(axis_name, expr, "inequality", other.group(1))
    other = re.fullmatch(
        rf"{NUMBER}\s*(?:<=|>=|<|>)\s*({_ANY_NAME}+)\s*(?:<=|>=|<|>)\s*{NUMBER}", text
    )
    if other:
        raise _mismatch(axis_name, expr, "inequality", other.group(1))
    if re.fullmatch(rf"{_ANY_NAME}*\s*=\s*{NUMBER}", text):
        raise ValueError(
            f"Invalid inequality expression: '{expr}' for axis '{axis_name}', "
            "use '==' to test for equality"
        )
    raise _mismatch(axis_name, expr, "inequality", None)


def _parse_categorical(axis_name: str, expr: str) -> tuple[str, str]:
    """Parses `name==value` / `name!=value` into (operator, value)."""
    text = expr.strip()
    name = re.escape(axis_name)

    match = re.fullmatch(rf"(?:{name})?\s*(==|!=)\s*({CATEGORY_VALUE})", text)
    if match:
        op, value = match.groups()
        if (
            value[0] in "'\""
            and value[-1] == value[0]
            and len(value) >= _MIN_QUOTED_LENGTH
        ):
            value = value[1:-1]
        return op, value

    other = re.fullmatch(rf"({_ANY_NAME}*)\s*(?:==|!=)\s*(?:{CATEGORY_VALUE})", text)
    if other and other.group(1):
        raise _mismatch(axis_name, expr, "categorical", other.group(1))
    raise _mismatch(axis_name, expr, "categorical", None)


def parse_categorical_to_mask(
    axis_name: str, expr: str, axis_points: np.ndarray
) -> np.ndarray:
    """
    Parse categorical equality expressions and return a boolean mask.

    Supports:
        - Equality: 'category==dog', 'label==positive'
        - Inequality: 'category!=dog'
        - Quoted values: "city=='new york'"
        - Without the axis name: '==dog'

    Args:
        axis_name: Name of the axis being filtered
        expr: Equality/inequality expression string
        axis_points: Array of categorical values for the axis

    Returns:
        Boolean mask array of same length as axis_points

    Raises:
        ValueError: If expression is invalid or variable name doesn't match axis

    Examples:
        >>> parse_categorical_to_mask('category', 'category==dog', np.array(['cat', 'dog', 'bird']))
        array([False,  True, False])

        >>> parse_categorical_to_mask('label', 'label!=positive', np.array(['positive', 'negative']))
        array([False,  True])
    """
    op, value = _parse_categorical(axis_name, expr)
    return _COMPARISONS[op](np.asarray(axis_points), value)


def parse_categorical_to_query(axis_name: str, expr: str) -> str:
    """
    Parse categorical equality expressions and return a pandas query string.

    Supports the same expressions as `parse_categorical_to_mask`.

    Args:
        axis_name: Name of the axis being filtered
        expr: Equality/inequality expression string

    Returns:
        Query string suitable for DataFrame.query()

    Raises:
        ValueError: If expression is invalid or variable name doesn't match axis

    Examples:
        >>> parse_categorical_to_query('category', 'category==dog')
        "`category` == 'dog'"

        >>> parse_categorical_to_query('label', 'label!=positive')
        "`label` != 'positive'"
    """
    op, value = _parse_categorical(axis_name, expr)
    return f"`{axis_name}` {op} {value!r}"


def parse_inequality_to_mask(
    axis_name: str, expr: str, axis_points: np.ndarray
) -> np.ndarray:
    """
    Parse inequality expressions for a given axis and return a boolean mask.

    Supports:
        - Simple comparisons: 'x<3', 'x>2', 'x>=5', 'x<=10', 'x==3', 'x!=3'
        - Compound inequalities: '-3<=x<3', '1<x<=5', '3>x>1'
        - Numbers like '1e-3' and '.5'
        - Without the axis name: '>2'

    Args:
        axis_name: Name of the axis being filtered
        expr: Inequality expression string
        axis_points: Array of coordinate values for the axis

    Returns:
        Boolean mask array of same length as axis_points

    Raises:
        ValueError: If expression is invalid or variable name doesn't match axis

    Examples:
        >>> parse_inequality_to_mask('x', 'x>2', np.array([0, 1, 2, 3, 4]))
        array([False, False, False,  True,  True])

        >>> parse_inequality_to_mask('e', '-3<=e<3', np.array([-5, -2, 0, 2, 5]))
        array([False,  True,  True,  True, False])
    """
    parsed = _parse_inequality(axis_name, expr)
    points = np.asarray(axis_points)
    if isinstance(parsed, _Compound):
        return compound_to_mask(
            points, float(parsed.value1), float(parsed.value2), parsed.op1, parsed.op2
        )
    return _COMPARISONS[parsed.op](points, float(parsed.value))


def parse_inequality_to_query(axis_name: str, expr: str) -> str:
    """
    Parse inequality expressions and return a pandas query string.

    Supports the same expressions as `parse_inequality_to_mask`.

    Args:
        axis_name: Name of the axis being filtered
        expr: Inequality expression string

    Returns:
        Query string suitable for DataFrame.query()

    Raises:
        ValueError: If expression is invalid or variable name doesn't match axis

    Examples:
        >>> parse_inequality_to_query('x', 'x>2')
        '`x` > 2'

        >>> parse_inequality_to_query('e', '-3<=e<3')
        '(-3 <= `e`) & (`e` < 3)'
    """
    parsed = _parse_inequality(axis_name, expr)
    if isinstance(parsed, _Compound):
        return (
            f"({parsed.value1} {parsed.op1} `{axis_name}`) & "
            f"(`{axis_name}` {parsed.op2} {parsed.value2})"
        )
    return f"`{axis_name}` {parsed.op} {parsed.value}"
