"""Axes of every type survive a save/load round trip, for DataBlock and Ensemble."""

import json

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import zarr

from NDLib.AxesStructures import CategoricalAxis, SignalAxis, UnorderedSignalAxis
from NDLib.BaseStructures import DataBlock, Ensemble


def _axes():
    return [
        SignalAxis(np.arange(3) * 1.0, "x", 0, "nm", navigate=True),
        UnorderedSignalAxis(np.array([5.0, 1.0]), "u", 1, "eV"),
        CategoricalAxis(["a", "b"], "param", 2, "-"),
    ]


def _assert_same_axes(loaded, expected):
    assert [type(ax) for ax in loaded] == [type(ax) for ax in expected]
    for got, want in zip(loaded, expected, strict=True):
        assert got == want
        assert (got.index_in_array, got.is_nav) == (want.index_in_array, want.is_nav)


def test_datablock_round_trip(tmp_path):
    raw = np.arange(12.0).reshape(3, 2, 2)
    block = DataBlock(da.from_array(raw, chunks=2), _axes(), "signal", "counts")

    block.save(tmp_path / "block")
    loaded = DataBlock.load(tmp_path / "block")

    _assert_same_axes(loaded.axes, block.axes)
    np.testing.assert_array_equal(loaded.compute(), raw)
    assert (loaded.quantity, loaded.unit) == ("signal", "counts")


def test_ensemble_round_trip_with_every_axis_type(tmp_path):
    x, u, p = np.meshgrid(np.arange(3.0), [5.0, 1.0], ["a", "b"], indexing="ij")
    frame = pd.DataFrame({
        "x": x.ravel(),
        "u": u.ravel(),
        "param": p.ravel(),
        "v": np.arange(12.0),
    })
    ensemble = Ensemble(dd.from_pandas(frame, npartitions=2), _axes(), "v", "counts")

    ensemble.save(tmp_path / "ensemble")
    loaded = Ensemble.load(tmp_path / "ensemble")

    _assert_same_axes(loaded.axes, ensemble.axes)
    assert (loaded.quantity, loaded.unit) == ("v", "counts")
    assert len(loaded.compute()) == 12


def test_unknown_axis_type_in_saved_metadata(tmp_path):
    block = DataBlock(
        da.from_array(np.arange(3.0), chunks=2), [_axes()[0]], "signal", "counts"
    )
    block.save(tmp_path / "block")
    root = zarr.open_group(str(tmp_path / "block.zarr"), mode="a")
    metadata = json.loads(str(root.attrs["metadata"]))
    metadata["axes"][0]["type"] = "MysteryAxis"
    root.attrs["metadata"] = json.dumps(metadata)

    with pytest.raises(ValueError, match="Unknown axis type: MysteryAxis"):
        DataBlock.load(tmp_path / "block")


@pytest.mark.parametrize(
    ("kind", "dropped", "message"),
    [
        ("SignalAxis", "points", "has no points"),
        ("UnorderedSignalAxis", "points", "has no points"),
        ("CategoricalAxis", "categories", "has no categories"),
    ],
)
def test_incomplete_axis_records(tmp_path, kind, dropped, message):
    axis = {
        "SignalAxis": _axes()[0],
        "UnorderedSignalAxis": _axes()[1],
        "CategoricalAxis": _axes()[2],
    }[kind]
    shape = (axis.size,)
    block = DataBlock(da.from_array(np.zeros(shape), chunks=2), [axis], "q", "-")
    block.save(tmp_path / "block")
    root = zarr.open_group(str(tmp_path / "block.zarr"), mode="a")
    metadata = json.loads(str(root.attrs["metadata"]))
    del metadata["axes"][0][dropped]
    root.attrs["metadata"] = json.dumps(metadata)

    with pytest.raises(ValueError, match=message):
        DataBlock.load(tmp_path / "block")
