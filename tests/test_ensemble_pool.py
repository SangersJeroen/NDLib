"""Tests for Ensemble.pool: pooling several axes by bin width."""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from NDLib.AxesStructures import CategoricalAxis, SignalAxis
from NDLib.BaseStructures import Ensemble


def _ensemble() -> Ensemble:
    x, y, rep = np.meshgrid(np.arange(10.0), np.arange(8.0), np.arange(2.0), indexing="ij")
    df = pd.DataFrame({
        "x": x.ravel(),
        "y": y.ravel(),
        "repetition": rep.ravel(),
        "v": x.ravel() + 10 * y.ravel() + 100 * rep.ravel(),
    })
    axes = [
        SignalAxis(np.arange(10.0), "x", 0, "nm", navigate=True),
        SignalAxis(np.arange(8.0), "y", 1, "nm", navigate=True),
        SignalAxis(np.arange(2.0), "repetition", 2, "#"),
    ]
    return Ensemble(dd.from_pandas(df, npartitions=3), axes, quantity="v", unit="counts")


def _frame(ensemble: Ensemble) -> pd.DataFrame:
    return ensemble.data.compute().sort_values(["x", "y", "repetition"]).reset_index(drop=True)


class TestEnsemblePool:
    def test_matches_chained_rebin_reduce_axis(self):
        ensemble = _ensemble()

        pooled = ensemble.pool({"x": 3.0, "y": 3.0})

        chained = ensemble.rebin_reduce_axis("x", new_binsize=3.0).rebin_reduce_axis(
            "y", new_binsize=3.0
        )
        pd.testing.assert_frame_equal(_frame(pooled), _frame(chained))
        assert [ax.name for ax in pooled.axes] == ["x", "y", "repetition"]
        assert (pooled.quantity, pooled.unit) == ("v", "counts")

    def test_other_axes_are_untouched(self):
        ensemble = _ensemble()

        pooled = ensemble.pool({"x": 3.0, "y": 3.0})

        frame = _frame(pooled)
        assert sorted(frame["repetition"].unique()) == [0.0, 1.0]
        np.testing.assert_array_equal(pooled.axes[2].points, np.arange(2.0))
        assert pooled.axes[0].size < 10
        assert pooled.axes[1].size < 8
        assert len(frame) == pooled.axes[0].size * pooled.axes[1].size * 2

    def test_custom_reducer(self):
        pooled = _ensemble().pool({"x": 3.0}, reducer=np.max)

        frame = _frame(pooled)
        # The largest x of each bin has the largest value, whatever y and repetition are
        expected = frame.groupby(["y", "repetition"])["v"].max()
        assert frame["v"].max() == expected.max()
        assert frame["v"].max() == 9 + 70 + 100

    def test_empty_bin_sizes_returns_an_equal_copy(self):
        ensemble = _ensemble()

        pooled = ensemble.pool({})

        assert pooled is not ensemble
        assert pooled.axes[0] is not ensemble.axes[0]
        pd.testing.assert_frame_equal(_frame(pooled), _frame(ensemble))

    def test_source_axes_are_not_modified(self):
        ensemble = _ensemble()

        ensemble.pool({"x": 3.0})

        assert ensemble.axes[0].size == 10

    def test_unknown_axis(self):
        with pytest.raises(RuntimeError, match="has no axis z"):
            _ensemble().pool({"z": 2.0})

    def test_categorical_axis(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "cat": ["a", "b"], "v": [1.0, 2.0]})
        axes = [
            SignalAxis(np.array([0.0, 1.0]), "x", 0, "nm"),
            CategoricalAxis(["a", "b"], "cat", 1, "-"),
        ]
        ensemble = Ensemble(dd.from_pandas(df, npartitions=1), axes, quantity="v")

        with pytest.raises(RuntimeError, match="CategoricalAxis"):
            ensemble.pool({"cat": 1.0})
