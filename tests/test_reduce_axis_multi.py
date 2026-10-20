"""reduce_axis over one or several axes for DataBlock and Ensemble."""

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from NDLib.AxesStructures import SignalAxis
from NDLib.BaseStructures import DataBlock, Ensemble

RAW = np.random.default_rng(0).random((6, 5, 4, 3))
NAMES = "abcd"


def _block(raw=RAW, chunks=(2, 2, 2, 2), *, computed=False) -> DataBlock:
    axes = [
        SignalAxis(np.arange(n) * 1.0, NAMES[i], i, "nm", navigate=True)
        for i, n in enumerate(raw.shape)
    ]
    data = raw if computed else da.from_array(raw, chunks=chunks[: raw.ndim])
    return DataBlock(data, axes, quantity="signal", unit="counts")


class TestDataBlockReduceAxis:
    def test_single_axis_default_reducer_is_max(self):
        result = _block().reduce_axis("b")

        np.testing.assert_allclose(result.compute(), RAW.max(axis=1))
        assert [ax.name for ax in result.axes] == ["a", "c", "d"]
        assert [ax.index_in_array for ax in result.axes] == [0, 1, 2]
        assert (result.quantity, result.unit) == ("q", "-")

    @pytest.mark.parametrize("names", [["b", "d"], ["d", "b"], ("b", "d")])
    def test_several_axes_see_all_values_at_once(self, names):
        result = _block().reduce_axis(names, reducer=np.median)

        # A median of medians would differ, so this checks the values are pooled
        np.testing.assert_allclose(result.compute(), np.median(RAW, axis=(1, 3)))
        assert [ax.name for ax in result.axes] == ["a", "c"]
        assert result.shape == (6, 4)

    def test_adjacent_and_outer_axes(self):
        np.testing.assert_allclose(
            _block().reduce_axis(["a", "b"], reducer=np.sum).compute(),
            RAW.sum(axis=(0, 1)),
        )
        np.testing.assert_allclose(
            _block().reduce_axis(["c", "d"], reducer=np.sum).compute(),
            RAW.sum(axis=(2, 3)),
        )

    def test_all_axes_gives_a_zero_dimensional_block(self):
        result = _block().reduce_axis(list(NAMES), reducer=np.median)

        assert result.axes == []
        assert result.shape == ()
        assert float(result.compute()) == pytest.approx(np.median(RAW))

    def test_output_quantity_and_unit(self):
        result = _block().reduce_axis(
            ["a", "b"], reducer=np.sum, out_quantity="total", out_unit="counts"
        )

        assert (result.quantity, result.unit) == ("total", "counts")

    def test_stays_lazy_and_source_is_untouched(self):
        db = _block()

        result = db.reduce_axis(["b", "d"], reducer=np.median)

        assert isinstance(result.data, da.Array)
        assert not result._computed
        assert [ax.name for ax in db.axes] == list(NAMES)
        assert [ax.index_in_array for ax in db.axes] == [0, 1, 2, 3]
        assert result.axes[0] is not db.axes[0]

    def test_axis_spread_over_many_chunks(self):
        raw = np.random.default_rng(1).random((4, 40))
        db = _block(raw, chunks=(2, 5))

        result = db.reduce_axis("b", reducer=np.median)

        np.testing.assert_allclose(result.compute(), np.median(raw, axis=1))

    def test_numpy_backed_block(self):
        result = _block(computed=True).reduce_axis(["a", "c"], reducer=np.sum)

        np.testing.assert_allclose(result.compute(), RAW.sum(axis=(0, 2)))

    def test_empty_axis_list(self):
        with pytest.raises(ValueError, match="at least one axis"):
            _block().reduce_axis([])

    def test_duplicate_axis(self):
        with pytest.raises(ValueError, match="unique"):
            _block().reduce_axis(["a", "a"])

    def test_unknown_axis(self):
        with pytest.raises(RuntimeError, match="Axis z not found"):
            _block().reduce_axis(["a", "z"])
        with pytest.raises(RuntimeError, match="Axis z not found"):
            _block().reduce_axis("z")


def _ensemble(*, as_pandas=False) -> Ensemble:
    a, b, c = np.meshgrid(np.arange(4.0), np.arange(3.0), np.arange(2.0), indexing="ij")
    df = pd.DataFrame({
        "a": a.ravel(),
        "b": b.ravel(),
        "c": c.ravel(),
        "v": a.ravel() + 10 * b.ravel() + 100 * c.ravel(),
    })
    axes = [
        SignalAxis(np.arange(4.0), "a", 0, "nm", navigate=True),
        SignalAxis(np.arange(3.0), "b", 1, "nm", navigate=True),
        SignalAxis(np.arange(2.0), "c", 2, "#"),
    ]
    data = df if as_pandas else dd.from_pandas(df, npartitions=3)
    return Ensemble(data, axes, quantity="v", unit="counts")


def _frame(ensemble: Ensemble, by) -> pd.DataFrame:
    data = ensemble.data
    frame = data.compute() if isinstance(data, dd.DataFrame) else data
    return frame.sort_values(by).reset_index(drop=True)


class TestEnsembleReduceAxis:
    @pytest.mark.parametrize("as_pandas", [False, True])
    def test_two_axes(self, as_pandas):
        result = _ensemble(as_pandas=as_pandas).reduce_axis(
            ["a", "c"], "sum", out_quantity="v"
        )

        frame = _frame(result, "b")
        # per b: sum over a=0..3 and c=0..1 of a + 10b + 100c
        expected = [(6 + 40 * b) * 2 + 100 * 4 for b in range(3)]
        assert list(frame["b"]) == [0.0, 1.0, 2.0]
        assert list(frame["v"]) == [float(e) for e in expected]
        assert [ax.name for ax in result.axes] == ["b"]
        assert result.axes[0].index_in_array == 0

    def test_data_column_is_renamed_to_out_quantity(self):
        result = _ensemble().reduce_axis(
            ["a", "b"], "max", out_quantity="peak", out_unit="a.u."
        )

        frame = _frame(result, "c")
        assert list(frame.columns) == ["c", "peak"]
        assert list(frame["peak"]) == [3 + 20.0, 3 + 20.0 + 100]
        assert (result.quantity, result.unit) == ("peak", "a.u.")
        # downstream operations can find the quantity column
        assert len(result.get({"c": "c>0"}).compute()) == 1

    def test_source_is_untouched(self):
        ensemble = _ensemble()

        ensemble.reduce_axis(["a"], "max")

        assert [ax.name for ax in ensemble.axes] == ["a", "b", "c"]
        assert [ax.index_in_array for ax in ensemble.axes] == [0, 1, 2]

    def test_cannot_reduce_all_axes(self):
        with pytest.raises(ValueError, match="Cannot reduce all axes"):
            _ensemble().reduce_axis(["a", "b", "c"], "max")

    def test_out_quantity_cannot_shadow_an_axis(self):
        with pytest.raises(ValueError, match="is the name of an axis"):
            _ensemble().reduce_axis("a", "max", out_quantity="b")

    def test_empty_duplicate_and_unknown_axes(self):
        with pytest.raises(ValueError, match="at least one axis"):
            _ensemble().reduce_axis([], "max")
        with pytest.raises(ValueError, match="unique"):
            _ensemble().reduce_axis(["a", "a"], "max")
        with pytest.raises(RuntimeError, match="Axis z not found"):
            _ensemble().reduce_axis(["a", "z"], "max")
