# pyright: basic
"""
Utilities for parsing indexing expressions in DataBlock and Ensemble.
"""

import re
import numpy as np

COMPOUND_PATTERN = (
# Pattern for compound inequalities: -3<=x<3 or 1<x<=5
    r"^([+-]?\d+\.?\d*)\s*([<>]=?)\s*(\w*\s*\w*)\s*([<>]=?)\s*([+-]?\d+\.?\d*)$"
)
# Pattern for simple inequalities: x>2, x<=5
SIMPLE_PATTERN = r"^(\w*\s*\w*)\s*([<=>]=?)\s*([+-]?\d+\.?\d*)$"


def parse_categorical_to_mask(axis_name: str, expr: str, axis_points: np.ndarray) -> np.ndarray:
    """
    Parse categorical equality expressions and return a boolean mask.

    Supports:
        - Equality: 'category==dog', 'label==positive'
        - Inequality: 'category!=dog'

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
    # Pattern for categorical equality/inequality: category==dog, label!=positive
    pattern = r"^([a-zA-Z_]\w*)\s*(==|!=)\s*([a-zA-Z_]\w*)$"

    match = re.match(pattern, expr.strip())
    if match:
        var_name, op, value = match.groups()
        if var_name != axis_name:
            raise ValueError(
                f"Variable '{var_name}' in expression doesn't match axis '{axis_name}'"
            )

        if op == "==":
            return axis_points == value
        else:  # '!='
            return axis_points != value

    raise ValueError(f"Invalid categorical expression: '{expr}' for axis '{axis_name}'")


def parse_categorical_to_query(axis_name: str, expr: str) -> str:
    """
    Parse categorical equality expressions and return a pandas query string.

    Supports:
        - Equality: 'category==dog', 'label==positive'
        - Inequality: 'category!=dog'

    Args:
        axis_name: Name of the axis being filtered
        expr: Equality/inequality expression string

    Returns:
        Query string suitable for DataFrame.query()

    Raises:
        ValueError: If expression is invalid or variable name doesn't match axis

    Examples:
        >>> parse_categorical_to_query('category', 'category==dog')
        "category == 'dog'"

        >>> parse_categorical_to_query('label', 'label!=positive')
        "label != 'positive'"
    """
    # Pattern for categorical equality/inequality: category==dog, label!=positive
    pattern = r"^([a-zA-Z_]\w*)\s*(==|!=)\s*([a-zA-Z_]\w*)$"

    match = re.match(pattern, expr.strip())
    if match:
        var_name, op, value = match.groups()
        if var_name != axis_name:
            raise ValueError(
                f"Variable '{var_name}' in expression doesn't match axis '{axis_name}'"
            )

        # Return query with properly quoted value
        return f"{axis_name} {op} '{value}'"

    raise ValueError(f"Invalid categorical expression: '{expr}' for axis '{axis_name}'")


def parse_inequality_to_mask(axis_name: str, expr: str, axis_points: np.ndarray) -> np.ndarray:
    """
    Parse inequality expressions for a given axis and return a boolean mask.

    Supports:
        - Simple inequalities: 'x<3', 'x>2', 'x>=5', 'x<=10'
        - Compound inequalities: '-3<=x<3', '1<x<=5'

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

    global COMPOUND_PATTERN
    global SIMPLE_PATTERN

    # Try compound inequality first
    compound_match = re.match(COMPOUND_PATTERN, expr.strip())
    if compound_match:
        val1, op1, var_name, op2, val2 = compound_match.groups()
        if var_name != axis_name:
            raise ValueError(
                f"Variable '{var_name}' in expression doesn't match axis '{axis_name}'"
            )

        val1, val2 = float(val1), float(val2)

        # Build compound mask
        if op1 == "<":
            mask1 = val1 < axis_points
        elif op1 == "<=":
            mask1 = val1 <= axis_points
        elif op1 == ">":
            mask1 = val1 > axis_points
        else:  # '>='
            mask1 = val1 >= axis_points

        if op2 == "<":
            mask2 = axis_points < val2
        elif op2 == "<=":
            mask2 = axis_points <= val2
        elif op2 == ">":
            mask2 = axis_points > val2
        else:  # '>='
            mask2 = axis_points >= val2

        return mask1 & mask2

    # Try simple inequality
    simple_match = re.match(SIMPLE_PATTERN, expr.strip())
    if simple_match:
        var_name, op, val = simple_match.groups()
        if var_name != axis_name:
            raise ValueError(
                f"Variable '{var_name}' in expression doesn't match axis '{axis_name}'"
            )

        val = float(val)

        if op == "<":
            return axis_points < val
        elif op == "<=":
            return axis_points <= val
        elif op == ">":
            return axis_points > val
        elif op == "==":
            return axis_points == val
        else:  # '>='
            return axis_points >= val

    raise ValueError(f"Invalid inequality expression: '{expr}' for axis '{axis_name}'")


def parse_inequality_to_query(axis_name: str, expr: str) -> str:
    """
    Parse inequality expressions and return a pandas query string.

    Supports:
        - Simple inequalities: 'x<3', 'x>2', 'x>=5', 'x<=10'
        - Compound inequalities: '-3<=x<3', '1<x<=5'

    Args:
        axis_name: Name of the axis being filtered
        expr: Inequality expression string

    Returns:
        Query string suitable for DataFrame.query()

    Raises:
        ValueError: If expression is invalid or variable name doesn't match axis

    Examples:
        >>> parse_inequality_to_query('x', 'x>2')
        'x > 2'

        >>> parse_inequality_to_query('e', '-3<=e<3')
        '(-3 <= e) & (e < 3)'
    """

    global COMPOUND_PATTERN
    global SIMPLE_PATTERN

    # Try compound inequality first
    compound_match = re.match(COMPOUND_PATTERN, expr.strip())
    if compound_match:
        val1, op1, var_name, op2, val2 = compound_match.groups()
        if var_name != axis_name:
            raise ValueError(
                f"Variable '{var_name}' in expression doesn't match axis '{axis_name}'"
            )

        # Build query string: val1 <= axis_name < val2
        return f"({val1} {op1} `{axis_name}`) & (`{axis_name}` {op2} {val2})"

    # Try simple inequality
    simple_match = re.match(SIMPLE_PATTERN, expr.strip())
    if simple_match:
        var_name, op, val = simple_match.groups()
        if var_name != axis_name:
            raise ValueError(
                f"Variable '{var_name}' in expression doesn't match axis '{axis_name}'"
            )
        return f"`{axis_name}` {op} {val}"

    raise ValueError(f"Invalid inequality expression: '{expr}' for axis '{axis_name}'")
