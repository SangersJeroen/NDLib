"""An Ensemble stays usable lazily after `compute()` turned its data into pandas."""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from NDLib.AxesStructures import SignalAxis
from NDLib.BaseStructures import DataBlock, Ensemble


def _ensemble(*, computed: bool, seed: int = 0) -> Ensemble:
    rng = np.random.default_rng(seed)
    x = SignalAxis(np.arange(10.0), "x", 0, "um")
    energy = SignalAxis(np.linspace(0, 100, 20), "E", 1, "eV")
    ensemble = DataBlock(rng.random((10, 20)), [x, energy], "q").to_ensemble()
    if computed:
        ensemble.compute()
    return ensemble


def _frame(ensemble: Ensemble) -> pd.DataFrame:
    data = ensemble.data
    frame = data.compute() if isinstance(data, dd.DataFrame) else data
    keys = [column for column in ("x", "E") if column in frame.columns]
    return frame.sort_values(keys).reset_index(drop=True)


def _labels() -> Ensemble:
    x = SignalAxis(np.arange(10.0), "x", 0, "um")
    labels = np.arange(10.0) % 3
    return DataBlock(labels, [x], "lab").to_ensemble()


OPERATIONS = {
    "rebin_reduce_axis": lambda e: e.rebin_reduce_axis("E", new_binsize=10.0),
    "pool": lambda e: e.pool({"E": 25.0}),
    "rebin_to_new_axis": lambda e: e.rebin_to_new_axis("E", new_binsize=10.0),
    "combine": lambda e: e.combine(
        _ensemble(computed=True, seed=1).rename_quantity("q2")
    ),
    "c_op": lambda e: e.c_op(_ensemble(computed=True, seed=1), lambda a, b: a - b),
}


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_matches_uncomputed_ensemble(name: str) -> None:
    operation = OPERATIONS[name]
    expected = _frame(operation(_ensemble(computed=False)))
    result = operation(_ensemble(computed=True))
    pd.testing.assert_frame_equal(_frame(result), expected)


@pytest.mark.parametrize("name", ["rebin_reduce_axis", "pool", "rebin_to_new_axis"])
def test_result_is_lazy(name: str) -> None:
    result = OPERATIONS[name](_ensemble(computed=True))
    assert isinstance(result.data, dd.DataFrame)


def test_data_is_not_replaced_by_helper() -> None:
    ensemble = _ensemble(computed=True)
    ensemble.pool({"E": 25.0})
    assert isinstance(ensemble.data, pd.DataFrame)


def test_to_datablock_after_compute() -> None:
    expected = _ensemble(computed=False).to_datablock().compute()
    result = _ensemble(computed=True).to_datablock().compute()
    np.testing.assert_array_equal(result, expected)


def test_split_with_ensemble_after_compute() -> None:
    expected = _ensemble(computed=False).split_with_ensemble(_labels(), ["x"])
    result = _ensemble(computed=True).split_with_ensemble(_labels(), ["x"])
    assert len(result) == len(expected) == 3
    for got, want in zip(result, expected, strict=True):
        pd.testing.assert_frame_equal(_frame(got), _frame(want))


def test_mixed_pandas_and_dask_operands() -> None:
    lazy = _ensemble(computed=False)
    computed = _ensemble(computed=True)
    assert len(_frame(lazy.combine(computed.rename_quantity("q2")))) == 200
    assert len(_frame(computed.combine(lazy.rename_quantity("q2")))) == 200
