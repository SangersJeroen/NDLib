"""Tests for DataBlock.pool: windowed reduction over one or more axes."""

import dask.array as da
import numpy as np
import pytest

from NDLib.AxesStructures import CategoricalAxis, SignalAxis, UnorderedSignalAxis
from NDLib.BaseStructures import DataBlock

RAW = np.random.default_rng(0).random((10, 8, 5))


def _block(raw: np.ndarray = RAW, names=("x", "y", "repetition")) -> DataBlock:
    axes = [
        SignalAxis(np.arange(n) * 0.5, name, i, "nm", navigate=True)
        for i, (n, name) in enumerate(zip(raw.shape, names))
    ]
    return DataBlock(
        da.from_array(raw, chunks=4), axes, quantity="intensity", unit="counts"
    )


def _block_means(raw: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Reference: mean over consecutive windows of `size`, trimming the remainder"""
    n = raw.shape[axis] // size * size
    moved = np.moveaxis(raw, axis, 0)[:n]
    return np.moveaxis(
        moved.reshape(n // size, size, *moved.shape[1:]).mean(axis=1), 0, axis
    )


class TestPoolValues:
    def test_pool_x_and_y_leaves_repetition_untouched(self):
        db = _block()

        result = db.pool({"x": 3, "y": 3})

        expected = _block_means(_block_means(RAW, 3, 0), 3, 1)
        assert result.shape == (3, 2, 5)
        assert [ax.name for ax in result.axes] == ["x", "y", "repetition"]
        np.testing.assert_allclose(result.compute(), expected)
        np.testing.assert_array_equal(result.axes[2].points, db.axes[2].points)
        assert [ax.index_in_array for ax in result.axes] == [0, 1, 2]
        assert (result.quantity, result.unit) == ("intensity", "counts")

    def test_new_points_are_window_means(self):
        result = _block().pool({"x": 3, "y": 3})

        np.testing.assert_allclose(result.axes[0].points, [0.5, 2.0, 3.5])
        np.testing.assert_allclose(result.axes[1].points, [0.5, 2.0])
        assert result.axes[0].size == 3
        assert result.axes[0].scale == pytest.approx(1.5)

    def test_result_stays_lazy(self):
        result = _block().pool({"x": 3})

        assert isinstance(result.data, da.Array)
        assert not result._computed

    def test_source_is_not_modified(self):
        db = _block()

        db.pool({"x": 3, "y": 3})

        assert db.shape == (10, 8, 5)
        assert db.axes[0].size == 10
        np.testing.assert_array_equal(db.compute(), RAW)

    def test_custom_reducer(self):
        result = _block().pool({"x": 2}, reducer=np.max)

        expected = np.moveaxis(RAW.reshape(5, 2, 8, 5).max(axis=1), 0, 0)
        np.testing.assert_allclose(result.compute(), expected)

    def test_single_axis(self):
        result = _block().pool({"y": 4})

        assert result.shape == (10, 2, 5)
        np.testing.assert_allclose(result.compute(), _block_means(RAW, 4, 1))

    @pytest.mark.parametrize("windows", [{}, {"x": 1}, {"x": 1, "y": 1}])
    def test_trivial_windows_return_an_equal_copy(self, windows):
        db = _block()

        result = db.pool(windows)

        assert result is not db
        assert result.axes[0] is not db.axes[0]
        np.testing.assert_array_equal(result.compute(), RAW)

    def test_numpy_backed_block_stays_numpy(self):
        axes = [SignalAxis(np.arange(6) * 1.0, "x", 0, "nm")]
        db = DataBlock(np.arange(6.0), axes)

        result = db.pool({"x": 2})

        assert isinstance(result.data, np.ndarray)
        np.testing.assert_allclose(result.compute(), [0.5, 2.5, 4.5])

    @pytest.mark.parametrize("shape", [(12,), (6, 4), (4, 6, 2, 3)])
    def test_other_dimensionalities(self, shape):
        raw = np.random.default_rng(1).random(shape)
        names = ("a", "b", "c", "d")[: len(shape)]
        db = _block(raw, names)

        result = db.pool({"a": 2})

        np.testing.assert_allclose(result.compute(), _block_means(raw, 2, 0))

    def test_unordered_axis_is_pooled_and_keeps_its_type(self):
        points = np.array([5.0, 1.0, 4.0, 2.0])
        axes = [UnorderedSignalAxis(points, "u", 0, "eV")]
        db = DataBlock(da.from_array(np.arange(4.0), chunks=2), axes)

        result = db.pool({"u": 2})

        assert isinstance(result.axes[0], UnorderedSignalAxis)
        np.testing.assert_allclose(result.axes[0].points, [3.0, 3.0])


class TestPoolRemainder:
    def test_trim_is_the_default(self):
        db = _block()

        assert db.pool({"x": 4}).shape == db.pool({"x": 4}, remainder="trim").shape
        assert db.pool({"x": 4}).shape == (2, 8, 5)

    def test_error_when_not_divisible(self):
        with pytest.raises(ValueError, match="not divisible by 3"):
            _block().pool({"x": 3}, remainder="error")

    def test_error_mode_accepts_divisible_axes(self):
        result = _block().pool({"x": 5}, remainder="error")

        assert result.shape == (2, 8, 5)
        np.testing.assert_allclose(result.compute(), _block_means(RAW, 5, 0))

    def test_pad_fills_last_window_with_nan(self):
        result = _block().pool({"x": 4}, reducer=np.nanmean, remainder="pad")

        assert result.shape == (3, 8, 5)
        np.testing.assert_allclose(result.compute()[:2], _block_means(RAW, 4, 0))
        np.testing.assert_allclose(result.compute()[2], RAW[8:].mean(axis=0))
        np.testing.assert_allclose(result.axes[0].points, [0.75, 2.75, 4.25])

    def test_pad_without_leftover_changes_nothing(self):
        result = _block().pool({"x": 5}, reducer=np.nanmean, remainder="pad")

        np.testing.assert_allclose(result.compute(), _block_means(RAW, 5, 0))

    def test_pad_with_plain_mean_gives_nan(self):
        result = _block().pool({"x": 4}, remainder="pad")

        assert np.isnan(result.compute()[2]).all()

    def test_pad_on_two_axes(self):
        result = _block().pool({"x": 4, "y": 3}, reducer=np.nanmean, remainder="pad")

        assert result.shape == (3, 3, 5)
        assert not np.isnan(result.compute()).any()

    def test_unknown_remainder(self):
        with pytest.raises(ValueError, match="remainder should be"):
            _block().pool({"x": 3}, remainder="wrap")


class TestPoolValidation:
    def test_unknown_axis(self):
        with pytest.raises(RuntimeError, match="Axis z not found"):
            _block().pool({"z": 2})

    def test_categorical_axis(self):
        axes = [
            SignalAxis(np.arange(4) * 1.0, "x", 0, "nm"),
            CategoricalAxis(["a", "b", "c"], "cat", 1, "-"),
        ]
        db = DataBlock(da.from_array(np.zeros((4, 3)), chunks=2), axes)

        with pytest.raises(RuntimeError, match="CategoricalAxis"):
            db.pool({"cat": 3})

    @pytest.mark.parametrize("window", [0, -2])
    def test_window_must_be_positive(self, window):
        with pytest.raises(ValueError, match=">= 1"):
            _block().pool({"x": window})

    @pytest.mark.parametrize("window", [2.0, "2", True, None])
    def test_window_must_be_an_int(self, window):
        with pytest.raises(ValueError, match="should be an int"):
            _block().pool({"x": window})

    def test_numpy_integer_windows_are_accepted(self):
        result = _block().pool({"x": np.int64(5)})

        assert result.shape == (2, 8, 5)

    def test_window_larger_than_axis_when_trimming(self):
        with pytest.raises(ValueError, match="larger than axis x"):
            _block().pool({"x": 11})

    def test_window_larger_than_axis_is_fine_when_padding(self):
        result = _block().pool({"x": 11}, reducer=np.nanmean, remainder="pad")

        # A single window leaves a size-1 axis, which DataBlock squeezes away
        assert result.shape == (8, 5)
        assert [ax.name for ax in result.axes] == ["y", "repetition"]
        np.testing.assert_allclose(result.compute(), RAW.mean(axis=0))
