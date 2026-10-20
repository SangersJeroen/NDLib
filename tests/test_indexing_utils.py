"""Grammar of the indexing expressions used by DataBlock.get and Ensemble.get."""

import numpy as np
import pytest

from NDLib.IndexingUtils import (
    compound_to_mask,
    parse_categorical_to_mask,
    parse_categorical_to_query,
    parse_inequality_to_mask,
    parse_inequality_to_query,
)

POINTS = np.array([-5.0, -2.0, 0.0, 0.5, 2.0, 5.0])


def _mask(expr, axis="x", points=POINTS):
    return parse_inequality_to_mask(axis, expr, points)


class TestSimpleInequalities:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("x>0", [0, 0, 0, 1, 1, 1]),
            ("x>=0", [0, 0, 1, 1, 1, 1]),
            ("x<0", [1, 1, 0, 0, 0, 0]),
            ("x<=0", [1, 1, 1, 0, 0, 0]),
            ("x==2", [0, 0, 0, 0, 1, 0]),
            ("x!=2", [1, 1, 1, 1, 0, 1]),
            ("x > 0", [0, 0, 0, 1, 1, 1]),
            ("  x<=-2  ", [1, 1, 0, 0, 0, 0]),
            ("x>.25", [0, 0, 0, 1, 1, 1]),
            ("x>1e0", [0, 0, 0, 0, 1, 1]),
            ("x<-1.5E0", [1, 1, 0, 0, 0, 0]),
            ("x>+0.5", [0, 0, 0, 0, 1, 1]),
            ("x<2.", [1, 1, 1, 1, 0, 0]),
        ],
    )
    def test_masks(self, expr, expected):
        np.testing.assert_array_equal(_mask(expr), np.array(expected, dtype=bool))

    @pytest.mark.parametrize(
        ("expr", "expected"), [(">0", [3, 4, 5]), ("<=-2", [0, 1])]
    )
    def test_axis_name_may_be_left_out(self, expr, expected):
        np.testing.assert_array_equal(np.nonzero(_mask(expr))[0], expected)

    def test_axis_names_with_special_characters(self):
        np.testing.assert_array_equal(
            _mask("E-loss<=0", axis="E-loss"), [True, True, True, False, False, False]
        )
        np.testing.assert_array_equal(
            _mask("k.x>1", axis="k.x"), [False, False, False, False, True, True]
        )

    def test_axis_named_like_an_exponent(self):
        np.testing.assert_array_equal(
            _mask("e<1e-3", axis="e"), [True, True, True, False, False, False]
        )


class TestCompoundInequalities:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("-3<=x<3", [0, 1, 1, 1, 1, 0]),
            ("-2<x<=2", [0, 0, 1, 1, 1, 0]),
            ("2>x>-2", [0, 0, 1, 1, 0, 0]),
            ("5>=x>=0.5", [0, 0, 0, 1, 1, 1]),
            (".5<x<1e1", [0, 0, 0, 0, 1, 1]),
        ],
    )
    def test_masks(self, expr, expected):
        np.testing.assert_array_equal(_mask(expr), np.array(expected, dtype=bool))

    def test_compound_to_mask_directly(self):
        mask = compound_to_mask(POINTS, -2, 2, "<=", "<")

        np.testing.assert_array_equal(mask, [False, True, True, True, False, False])

    @pytest.mark.parametrize("expr", ["1<x>3", "3>x<1"])
    def test_mixed_directions_are_rejected(self, expr):
        with pytest.raises(ValueError, match="both sides"):
            _mask(expr)


class TestInequalityErrors:
    def test_wrong_variable_name(self):
        with pytest.raises(
            ValueError, match="Variable 'y' in expression doesn't match"
        ):
            _mask("y>1")

    def test_wrong_variable_name_in_compound(self):
        with pytest.raises(
            ValueError, match="Variable 'y' in expression doesn't match"
        ):
            _mask("1<y<3")

    def test_single_equals_is_rejected(self):
        with pytest.raises(ValueError, match="use '=='"):
            _mask("x=1")

    @pytest.mark.parametrize("expr", ["", "x", "x>", "x>>1", "x ~ 3", "x>one", "x>1>2"])
    def test_invalid_expressions(self, expr):
        with pytest.raises(ValueError, match="Invalid inequality expression"):
            _mask(expr)


class TestInequalityQueries:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("x>2", "`x` > 2"),
            (">2", "`x` > 2"),
            ("x != .5", "`x` != .5"),
            ("x<=1e-3", "`x` <= 1e-3"),
            ("-3<=x<3", "(-3 <= `x`) & (`x` < 3)"),
            ("3>x>1", "(3 > `x`) & (`x` > 1)"),
        ],
    )
    def test_query_strings(self, expr, expected):
        assert parse_inequality_to_query("x", expr) == expected

    def test_axis_names_are_backtick_quoted(self):
        assert parse_inequality_to_query("E-loss", "E-loss>1") == "`E-loss` > 1"

    def test_errors_match_the_mask_parser(self):
        with pytest.raises(ValueError, match="doesn't match"):
            parse_inequality_to_query("x", "y>1")
        with pytest.raises(ValueError, match="use '=='"):
            parse_inequality_to_query("x", "x=1")
        with pytest.raises(ValueError, match="both sides"):
            parse_inequality_to_query("x", "1<x>3")


CATEGORIES = np.array(["cat", "dog", "new york", "a-b", "3"])


class TestCategorical:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("c==dog", [0, 1, 0, 0, 0]),
            ("c!=dog", [1, 0, 1, 1, 1]),
            ("c == dog", [0, 1, 0, 0, 0]),
            ("==dog", [0, 1, 0, 0, 0]),
            ("c=='new york'", [0, 0, 1, 0, 0]),
            ('c=="new york"', [0, 0, 1, 0, 0]),
            ("c==a-b", [0, 0, 0, 1, 0]),
            ("c==3", [0, 0, 0, 0, 1]),
            ("c==''", [0, 0, 0, 0, 0]),
        ],
    )
    def test_masks(self, expr, expected):
        mask = parse_categorical_to_mask("c", expr, CATEGORIES)

        np.testing.assert_array_equal(mask, np.array(expected, dtype=bool))

    def test_queries(self):
        assert parse_categorical_to_query("c", "c==dog") == "`c` == 'dog'"
        assert parse_categorical_to_query("c", "c!=dog") == "`c` != 'dog'"
        assert parse_categorical_to_query("c", "c=='new york'") == "`c` == 'new york'"
        assert parse_categorical_to_query("c", "==dog") == "`c` == 'dog'"

    def test_wrong_variable_name(self):
        with pytest.raises(
            ValueError, match="Variable 'd' in expression doesn't match"
        ):
            parse_categorical_to_mask("c", "d==dog", CATEGORIES)
        with pytest.raises(ValueError, match="doesn't match"):
            parse_categorical_to_query("c", "d==dog")

    @pytest.mark.parametrize("expr", ["", "c", "c=dog", "c>dog", "c==", "c==two words"])
    def test_invalid_expressions(self, expr):
        with pytest.raises(ValueError, match="Invalid categorical expression"):
            parse_categorical_to_mask("c", expr, CATEGORIES)
        with pytest.raises(ValueError, match="Invalid categorical expression"):
            parse_categorical_to_query("c", expr)
