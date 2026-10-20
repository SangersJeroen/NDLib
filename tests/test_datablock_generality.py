"""Tests for DataBlock behaviour on N-dimensional data: squeezing, iteration, arithmetic."""

import dask.array as da
import numpy as np
import pytest

from NDLib.AxesStructures import CategoricalAxis, SignalAxis
from NDLib.BaseStructures import DataBlock


def _block(shape=(4, 3, 2), *, computed=False, quantity="intensity", unit="counts"):
    raw = np.arange(float(np.prod(shape))).reshape(shape)
    names = ("x", "y", "z", "w")[: len(shape)]
    axes = [
        SignalAxis(np.arange(n) * 1.0, name, i, "nm", navigate=True)
        for i, (n, name) in enumerate(zip(shape, names, strict=True))
    ]
    data = raw if computed else da.from_array(raw, chunks=2)
    return DataBlock(data, axes, quantity=quantity, unit=unit), raw


class TestSqueezeKeepsQuantity:
    def test_numeric_size_one_axis_keeps_quantity(self):
        db, raw = _block()

        cropped = db.crop_axis("x", 0.0, 0.0)

        assert cropped.shape == (3, 2)
        assert (cropped.quantity, cropped.unit) == ("intensity", "counts")
        assert [ax.name for ax in cropped.axes] == ["y", "z"]
        assert [ax.index_in_array for ax in cropped.axes] == [0, 1]
        np.testing.assert_array_equal(cropped.compute(), raw[0])

    def test_pooling_to_a_single_window_keeps_quantity(self):
        db, raw = _block((4, 3, 2))

        pooled = db.pool({"x": 4})

        assert pooled.shape == (3, 2)
        assert (pooled.quantity, pooled.unit) == ("intensity", "counts")
        np.testing.assert_allclose(pooled.compute(), raw.mean(axis=0))

    def test_categorical_size_one_axis_still_renames_quantity(self):
        axes = [
            SignalAxis(np.arange(4) * 1.0, "x", 0, "nm"),
            CategoricalAxis(["amplitude"], "parameter", 1, "-"),
        ]

        db = DataBlock(da.from_array(np.zeros((4, 1)), chunks=2), axes, quantity="fit")

        assert db.shape == (4,)
        assert db.quantity == "parameter"

    def test_axes_are_renumbered_without_a_warning(self, caplog):
        db, _ = _block()

        with caplog.at_level("WARNING"):
            db.crop_axis("y", 1.0, 1.0)

        assert "fixed axis index" not in caplog.text


class TestIterAxis:
    @pytest.mark.parametrize("shape", [(5, 4), (5, 4, 3), (3, 4, 2, 5)])
    def test_every_axis_of_every_dimensionality(self, shape):
        db, raw = _block(shape)

        for axis_index, axis in enumerate(db.axes):
            slices = list(db.iter_axis(axis.name))

            assert len(slices) == shape[axis_index]
            for i, piece in enumerate(slices):
                np.testing.assert_array_equal(
                    np.asarray(piece), np.take(raw, i, axis=axis_index)
                )

    def test_min_and_max_idx(self):
        db, raw = _block((6, 3, 2))

        slices = list(db.iter_axis("x", min_idx=2, max_idx=5))

        assert len(slices) == 3
        np.testing.assert_array_equal(np.asarray(slices[0]), raw[2])

    def test_only_min_idx(self):
        db, raw = _block((6, 3, 2))

        slices = list(db.iter_axis("x", min_idx=4))

        assert len(slices) == 2
        np.testing.assert_array_equal(np.asarray(slices[-1]), raw[5])

    def test_only_max_idx(self):
        db, raw = _block((6, 3, 2))

        slices = list(db.iter_axis("y", max_idx=2))

        assert len(slices) == 2
        np.testing.assert_array_equal(np.asarray(slices[1]), raw[:, 1])

    def test_source_not_modified(self):
        db, raw = _block((6, 3, 2))

        list(db.iter_axis("z"))

        np.testing.assert_array_equal(db.compute(), raw)


class TestArithmetic:
    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            (lambda db: db + 2, lambda raw: raw + 2),
            (lambda db: 2 + db, lambda raw: 2 + raw),
            (lambda db: db - 2, lambda raw: raw - 2),
            (lambda db: 2 - db, lambda raw: 2 - raw),
            (lambda db: db * 3, lambda raw: raw * 3),
            (lambda db: 3 * db, lambda raw: 3 * raw),
            (lambda db: db / 2, lambda raw: raw / 2),
            (lambda db: 2 / (db + 1), lambda raw: 2 / (raw + 1)),
            (lambda db: db**2, lambda raw: raw**2),
            (lambda db: -db, lambda raw: -raw),
        ],
        ids=["add", "radd", "sub", "rsub", "mul", "rmul", "div", "rdiv", "pow", "neg"],
    )
    def test_scalar_operations(self, operation, expected):
        db, raw = _block()

        result = operation(db)

        assert isinstance(result, DataBlock)
        np.testing.assert_allclose(result.compute(), expected(raw))
        assert (result.quantity, result.unit) == ("intensity", "counts")
        assert [ax.name for ax in result.axes] == ["x", "y", "z"]

    def test_result_owns_its_axes(self):
        db, _ = _block()

        result = db + 1

        assert result.axes[0] is not db.axes[0]
        result.axes[0].name = "renamed"
        assert db.axes[0].name == "x"

    def test_two_datablocks(self):
        a, raw = _block()
        b, _ = _block()

        np.testing.assert_allclose((a + b).compute(), raw + raw)
        np.testing.assert_allclose((a - b).compute(), np.zeros_like(raw))
        np.testing.assert_allclose((a * b).compute(), raw * raw)
        np.testing.assert_allclose((a / (b + 1)).compute(), raw / (raw + 1))

    def test_broadcastable_array(self):
        db, raw = _block()
        offsets = np.array([10.0, 20.0])

        result = db + offsets

        np.testing.assert_allclose(result.compute(), raw + offsets)

    def test_mismatched_axis_points(self):
        a, _ = _block()
        axes = [
            SignalAxis(np.arange(n) * 2.0, name, i, "nm", navigate=True)
            for i, (n, name) in enumerate(zip((4, 3, 2), ("x", "y", "z"), strict=True))
        ]
        b = DataBlock(da.from_array(np.ones((4, 3, 2)), chunks=2), axes)

        with pytest.raises(ValueError, match="identical axes"):
            a + b

    def test_mismatched_axis_names_or_count(self):
        a, _ = _block((4, 3))
        b, _ = _block((4, 3, 2))
        renamed, _ = _block((4, 3))
        renamed.axes[1].name = "other"

        with pytest.raises(ValueError, match="identical axes"):
            a + b
        with pytest.raises(ValueError, match="identical axes"):
            a + renamed

    def test_numpy_backed_block_stays_numpy(self):
        db, raw = _block(computed=True)

        result = db * 2

        assert isinstance(result.data, np.ndarray)
        np.testing.assert_allclose(result.compute(), raw * 2)
