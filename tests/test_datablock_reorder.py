"""reorder_axis returns a new block, reorder_axis_inplace changes the instance itself."""

import dask.array as da
import numpy as np
import pytest

from NDLib.AxesStructures import SignalAxis
from NDLib.BaseStructures import DataBlock

RAW = np.arange(24, dtype=float).reshape(2, 3, 4)


def _block(*, computed=False) -> DataBlock:
    axes = [
        SignalAxis(np.arange(n) * 1.0, name, i, "nm", navigate=True)
        for i, (n, name) in enumerate([(2, "a"), (3, "b"), (4, "c")])
    ]
    data = RAW.copy() if computed else da.from_array(RAW, chunks=2)
    return DataBlock(data, axes, quantity="signal", unit="counts")


class TestReorderAxisReturnsNewBlock:
    @pytest.mark.parametrize(
        ("axis_name", "target", "names", "source"),
        [
            ("c", 0, ["c", "a", "b"], 2),
            ("a", -1, ["b", "c", "a"], 0),
            ("b", 2, ["a", "c", "b"], 1),
        ],
    )
    def test_new_block_has_moved_axis_and_data(self, axis_name, target, names, source):
        db = _block()

        result = db.reorder_axis(axis_name, target)

        assert result is not db
        assert [ax.name for ax in result.axes] == names
        assert [ax.index_in_array for ax in result.axes] == [0, 1, 2]
        np.testing.assert_array_equal(
            result.compute(), np.moveaxis(RAW, source, target)
        )
        assert (result.quantity, result.unit) == ("signal", "counts")

    def test_source_is_untouched(self):
        db = _block()

        db.reorder_axis("c", 0)

        assert [ax.name for ax in db.axes] == ["a", "b", "c"]
        assert [ax.index_in_array for ax in db.axes] == [0, 1, 2]
        assert db.shape == (2, 3, 4)
        np.testing.assert_array_equal(db.compute(), RAW)

    def test_axes_are_copies(self):
        db = _block()

        result = db.reorder_axis("c", 0)

        assert result.axis_obj("c") is not db.axis_obj("c")
        assert result.axis_obj("a").index_in_array == 1
        assert db.axis_obj("a").index_in_array == 0

    def test_same_position_returns_an_equal_copy(self):
        db = _block()

        result = db.reorder_axis("b", 1)

        assert result is not db
        assert result.axes[1] is not db.axes[1]
        assert [ax.name for ax in result.axes] == ["a", "b", "c"]
        np.testing.assert_array_equal(result.compute(), RAW)

    def test_lazy_and_numpy_backends_are_preserved(self):
        lazy = _block().reorder_axis("c", 0)
        eager = _block(computed=True).reorder_axis("c", 0)

        assert isinstance(lazy.data, da.Array)
        assert not lazy._computed
        assert isinstance(eager.data, np.ndarray)
        assert eager._computed

    def test_chaining(self):
        result = _block().reorder_axis("c", 0).reorder_axis("a", 2)

        assert [ax.name for ax in result.axes] == ["c", "b", "a"]
        np.testing.assert_array_equal(
            result.compute(), np.moveaxis(np.moveaxis(RAW, 2, 0), 1, 2)
        )

    def test_unknown_axis(self):
        with pytest.raises(RuntimeError, match="no axis: z"):
            _block().reorder_axis("z", 0)

    @pytest.mark.parametrize("target", [3, -4])
    def test_target_out_of_range(self, target):
        with pytest.raises(ValueError, match="out of range"):
            _block().reorder_axis("a", target)


class TestReorderAxisInplaceMatchesReorderAxis:
    @pytest.mark.parametrize(("axis_name", "target"), [("c", 0), ("a", -1), ("b", 1)])
    def test_same_result_as_the_new_block_version(self, axis_name, target):
        expected = _block().reorder_axis(axis_name, target)
        db = _block()

        returned = db.reorder_axis_inplace(axis_name, target)

        assert returned is None
        assert [ax.name for ax in db.axes] == [ax.name for ax in expected.axes]
        assert db.shape == expected.shape
        np.testing.assert_array_equal(db.compute(), expected.compute())
