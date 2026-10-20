"""Axes compare by value and are hashable, so they can live in sets and dicts."""

from copy import deepcopy

import numpy as np

from NDLib.AxesStructures import Axis, CategoricalAxis, SignalAxis, UnorderedSignalAxis


def _signal(points=(0.0, 1.0, 2.0), name="x", unit="nm", index=0, **kwargs):
    return SignalAxis(np.array(points), name, index, unit, **kwargs)


def _unordered(points=(5.0, 1.0, 3.0), name="u", unit="eV", index=0):
    return UnorderedSignalAxis(np.array(points), name, index, unit)


def _categorical(categories=("a", "b"), name="c", unit="-", index=0):
    return CategoricalAxis(list(categories), name, index, unit)


class TestHashable:
    def test_all_axis_types_can_be_hashed(self):
        for axis in (_signal(), _unordered(), _categorical()):
            assert isinstance(hash(axis), int)

    def test_set_and_dict_membership(self):
        axes = {_signal(), _unordered(), _categorical()}

        assert len(axes) == 3
        assert _signal() in axes
        assert _categorical() in axes
        lookup = {_signal(): "x axis"}
        assert lookup[_signal()] == "x axis"

    def test_duplicates_collapse(self):
        assert len({_signal(), _signal(), deepcopy(_signal())}) == 1

    def test_deepcopy_is_equal_with_equal_hash(self):
        for axis in (_signal(), _unordered(), _categorical()):
            copy = deepcopy(axis)

            assert copy == axis
            assert hash(copy) == hash(axis)

    def test_layout_does_not_affect_hash_or_equality(self):
        axis = _signal(index=0, navigate=False)
        moved = _signal(index=3, navigate=True)

        assert axis == moved
        assert hash(axis) == hash(moved)

    def test_int_and_float_points_with_equal_values_agree(self):
        ints = _signal(points=(0, 1, 2))
        floats = _signal(points=(0.0, 1.0, 2.0))

        assert ints == floats
        assert hash(ints) == hash(floats)

    def test_negative_zero(self):
        assert hash(_signal(points=(-0.0, 1.0))) == hash(_signal(points=(0.0, 1.0)))

    def test_string_widths_do_not_matter(self):
        short = _categorical(("a", "b"))
        long = CategoricalAxis(np.array(["a", "b"], dtype="U10").tolist(), "c", 0, "-")

        assert short == long
        assert hash(short) == hash(long)


class TestEqualityBetweenAxes:
    def test_returns_a_plain_bool(self):
        assert (_signal() == _signal()) is True
        assert (_signal() == _signal(points=(0.0, 1.0, 5.0))) is False
        assert (_categorical() == _categorical()) is True

    def test_differences_make_axes_unequal(self):
        base = _signal()

        assert base != _signal(points=(0.0, 1.0, 2.5))
        assert base != _signal(points=(0.0, 1.0))
        assert base != _signal(name="y")
        assert base != _signal(unit="um")

    def test_different_types_are_unequal(self):
        assert _signal(points=(1.0, 2.0, 5.0)) != _unordered(
            points=(1.0, 2.0, 5.0), name="x", unit="nm"
        )

    def test_unordered_axes(self):
        assert _unordered() == _unordered()
        assert _unordered() != _unordered(points=(1.0, 3.0, 5.0))

    def test_base_axis_without_points(self):
        assert Axis("a", 0, "-") == Axis("a", 0, "-")
        assert hash(Axis("a", 0, "-")) == hash(Axis("a", 1, "-"))


class TestElementwiseComparisonIsUnchanged:
    def test_signal_axis_against_scalar_and_array(self):
        axis = _signal()

        np.testing.assert_array_equal(axis == 1.0, [False, True, False])
        np.testing.assert_array_equal(
            axis == np.array([0.0, 5.0, 2.0]), [True, False, True]
        )
        np.testing.assert_array_equal(axis > 0.5, [False, True, True])
        np.testing.assert_array_equal(axis <= 1.0, [True, True, False])

    def test_categorical_axis_against_string(self):
        np.testing.assert_array_equal(_categorical() == "b", [False, True])

    def test_unordered_axis_has_no_elementwise_comparison(self):
        assert (_unordered() == 3.0) is False
