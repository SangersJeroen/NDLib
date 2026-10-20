"""SignalAxis only accepts 1-D, strictly increasing points."""

import numpy as np
import pytest

from NDLib.AxesStructures import SignalAxis, UnorderedSignalAxis


def _axis(points):
    return SignalAxis(points, "x", 0, "nm")


class TestAccepted:
    @pytest.mark.parametrize(
        "points",
        [[0.0, 1.0, 2.0], [-3, -1, 4], [1e-9, 2e-9], np.linspace(0, 1, 50), (1.0, 2.0)],
    )
    def test_strictly_increasing(self, points):
        axis = _axis(points)

        assert axis.size == len(points)
        assert axis.ordered

    @pytest.mark.parametrize("points", [[], [4.0]])
    def test_fewer_than_two_points(self, points):
        axis = _axis(points)

        assert axis.size == len(points)
        assert axis.scale == np.inf

    def test_scale_min_max_of_a_valid_axis(self):
        axis = _axis([1.0, 3.0, 7.0])

        assert (axis.scale, axis.min, axis.max) == (2.0, 1.0, 7.0)


class TestRejected:
    @pytest.mark.parametrize(
        "points",
        [
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0],
            [0.0, np.nan, 2.0],
            [np.nan, np.nan],
        ],
        ids=["unsorted", "descending", "duplicate", "all-equal", "nan", "all-nan"],
    )
    def test_not_strictly_increasing(self, points):
        with pytest.raises(ValueError, match="strictly increasing"):
            _axis(points)

    def test_message_names_the_axis_and_the_alternative(self):
        with pytest.raises(ValueError, match=r"SignalAxis 'x'.*UnorderedSignalAxis"):
            _axis([2.0, 1.0])

    def test_not_one_dimensional(self):
        with pytest.raises(ValueError, match="should be 1-D, got 2 dimensions"):
            _axis(np.arange(6.0).reshape(2, 3))

    def test_unordered_axis_accepts_any_order(self):
        axis = UnorderedSignalAxis(np.array([3.0, 1.0, 2.0, 2.0]), "x", 0, "nm")

        assert axis.size == 4
