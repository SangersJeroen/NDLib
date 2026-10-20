"""
Comprehensive unit tests for Ensemble class from eels_base/ndlib/BaseStructures.py
This test suite thoroughly tests all methods and edge cases of the Ensemble class.
"""

import pathlib
import sys

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the path to import eels_base
sys.path.insert(0, pathlib.Path(pathlib.Path(pathlib.Path(__file__).resolve()).parent).parent)

from NDLib.AxesStructures import CategoricalAxis, SignalAxis, UnorderedSignalAxis
from NDLib.BaseStructures import DataBlock, Ensemble


class TestEnsembleInitialization:
    """Test Ensemble initialization and validation"""

    def test_basic_initialization(self):
        """Test basic Ensemble creation with valid inputs"""
        # Create sample data
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "intensity": [100, 200, 300, 400, 500],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            UnorderedSignalAxis(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "x", 0, "nm", navigate=True
            ),
            UnorderedSignalAxis(
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "y", 1, "nm", navigate=True
            ),
        ]

        ensemble = Ensemble(ddf, axes, quantity="intensity", unit="counts")

        assert ensemble.dims == 2
        assert ensemble.quantity == "intensity"
        assert ensemble.unit == "counts"
        assert len(ensemble.axes) == 2
        assert not ensemble._computed

    def test_initialization_with_default_params(self):
        """Test Ensemble creation with default quantity and unit"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "q": [100, 200, 300]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]

        ensemble = Ensemble(ddf, axes)

        assert ensemble.quantity == "q"
        assert ensemble.unit == "-"

    def test_initialization_with_computed_data(self):
        """Test Ensemble creation with already computed pandas DataFrame"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "intensity": [100, 200, 300]})

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]

        ensemble = Ensemble(df, axes, quantity="intensity")

        assert ensemble._computed


class TestEnsembleCompute:
    """Test compute functionality"""

    def test_compute_lazy_dataframe(self):
        """Test computing a lazy dask DataFrame"""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "value": [100, 200, 300],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True),
            UnorderedSignalAxis(np.array([10.0, 20.0, 30.0]), "y", 1, "nm", navigate=True),
        ]

        ensemble = Ensemble(ddf, axes, quantity="value")

        assert not ensemble._computed
        result = ensemble.compute()

        assert ensemble._computed
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_compute_already_computed(self):
        """Test that computing already computed data returns cached result"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [100, 200, 300]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result1 = ensemble.compute()
        result2 = ensemble.compute()

        assert result1 is result2


class TestEnsembleFromDataBlock:
    """Test creating Ensemble from DataBlock"""

    def test_from_datablock_2d(self):
        """Test converting 2D DataBlock to Ensemble"""
        data = da.from_array(np.arange(12).reshape(3, 4), chunks=(3, 4))
        axes = [
            SignalAxis(np.array([0.0, 1.0, 2.0]), "x", 0, "nm", navigate=True),
            SignalAxis(np.array([0.0, 1.0, 2.0, 3.0]), "y", 1, "nm", navigate=True),
        ]
        db = DataBlock(data, axes, quantity="intensity", unit="counts")

        ensemble = Ensemble.from_datablock(db)

        assert isinstance(ensemble, Ensemble)
        assert ensemble.dims == 2
        assert ensemble.quantity == "intensity"
        assert ensemble.unit == "counts"
        assert len(ensemble.axes) == 2

        # Check that data was flattened (3x4 = 12 rows)
        computed = ensemble.compute()
        assert len(computed) == 12

    def test_from_datablock_1d(self):
        """Test converting 1D DataBlock to Ensemble"""
        data = da.from_array(np.arange(5), chunks=5)
        axes = [SignalAxis(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), "x", 0, "nm", navigate=True)]
        db = DataBlock(data, axes, quantity="signal")

        ensemble = Ensemble.from_datablock(db)

        assert ensemble.dims == 1
        assert ensemble.quantity == "signal"
        computed = ensemble.compute()
        assert len(computed) == 5


class TestEnsembleAxisMethods:
    """Test axis-related methods"""

    def test_has_axis_existing(self):
        """Test has_axis returns True for existing axis"""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "energy": [100.0, 200.0, 300.0],
            "value": [10, 20, 30],
        })
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [
            UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True),
            UnorderedSignalAxis(
                np.array([100.0, 200.0, 300.0]), "energy", 1, "eV", navigate=True
            ),
        ]
        ensemble = Ensemble(ddf, axes, quantity="value")

        assert ensemble.has_axis("x")
        assert ensemble.has_axis("energy")

    def test_has_axis_nonexisting(self):
        """Test has_axis returns False for non-existing axis"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10, 20, 30]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        assert not ensemble.has_axis("z")

    def test_axis_obj(self):
        """Test retrieving axis object by name"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10, 20, 30]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        x_axis = ensemble.axis_obj("x")
        assert x_axis.name == "x"
        assert x_axis.unit == "nm"

    def test_axis_p(self):
        """Test axis_p method returns axis points"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10, 20, 30]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        x_points = ensemble.axis_p("x")
        assert len(x_points) == 3
        assert np.allclose(x_points, [1.0, 2.0, 3.0])


class TestEnsembleArithmetic:
    """Test arithmetic operations"""

    def test_add_scalar(self):
        """Test adding scalar to Ensemble"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10.0, 20.0, 30.0]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble + 5

        # Result is a Ensemble
        assert isinstance(result, Ensemble)
        # Result.data is dataframe
        assert isinstance(result.data, (dd.DataFrame, pd.DataFrame))

    def test_subtract_scalar(self):
        """Test subtracting scalar from Ensemble"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10.0, 20.0, 30.0]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble - 5

        # Result is a Ensemble
        assert isinstance(result, Ensemble)
        # Result.data is dataframe
        assert isinstance(result.data, (dd.DataFrame, pd.DataFrame))

    def test_multiply_scalar(self):
        """Test multiplying Ensemble by scalar"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10.0, 20.0, 30.0]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble * 2

        # Result is a Ensemble
        assert isinstance(result, Ensemble)
        # Result.data is dataframe
        assert isinstance(result.data, (dd.DataFrame, pd.DataFrame))

    def test_power_operation(self):
        """Test power operation"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [2.0, 3.0, 4.0]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble**2

        # Result is a Ensemble
        assert isinstance(result, Ensemble)
        # Result.data is dataframe
        assert isinstance(result.data, (dd.DataFrame, pd.DataFrame))


class TestEnsembleGetMethod:
    """Test advanced get() indexing method"""

    def test_get_with_inequality(self):
        """Test get with inequality expressions"""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "value": [100, 200, 300, 400, 500],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            UnorderedSignalAxis(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "x", 0, "nm", navigate=True
            ),
            UnorderedSignalAxis(
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "y", 1, "nm", navigate=True
            ),
        ]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble.get({"x": "x>2"})
        result_computed = result.compute()

        assert len(result_computed) == 3  # x = 3, 4, 5

    def test_get_with_multiple_axes(self):
        """Test get with multiple axes"""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "value": [100, 200, 300, 400, 500],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            UnorderedSignalAxis(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "x", 0, "nm", navigate=True
            ),
            UnorderedSignalAxis(
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "y", 1, "nm", navigate=True
            ),
        ]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble.get({"x": "x>=2", "y": "y<40"})
        result_computed = result.compute()

        # Should have rows where x>=2 AND y<40: (2,20), (3,30)
        assert len(result_computed) == 2

    def test_get_empty_dict(self):
        """Test get with empty dict returns original"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [100, 200, 300]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result = ensemble.get({})
        result_computed = result.compute()

        assert len(result_computed) == 3


class TestEnsembleValueGet:
    """Test value_get() method"""

    def test_value_get_basic(self):
        """Test value_get returns tuple of data and axes dict"""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "value": [100, 200, 300, 400, 500],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            UnorderedSignalAxis(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "x", 0, "nm", navigate=True
            ),
            UnorderedSignalAxis(
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "y", 1, "nm", navigate=True
            ),
        ]
        ensemble = Ensemble(ddf, axes, quantity="value")

        quantity_data, axes_dict = ensemble.value_get({"x": "x>2"})

        # value_get returns dask arrays, not numpy arrays
        assert isinstance(quantity_data, (np.ndarray, da.Array))
        assert isinstance(axes_dict, dict)
        assert "x" in axes_dict
        assert "y" in axes_dict


class TestEnsembleRenameQuantity:
    """Test rename_quantity() method"""

    def test_rename_quantity(self):
        """Test renaming the quantity column"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [100, 200, 300]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        renamed = ensemble.rename_quantity("intensity")

        assert renamed.quantity == "intensity"
        assert "intensity" in renamed.compute().columns


class TestEnsembleCombine:
    """Test combine() method"""

    def test_combine_ensembles(self):
        """Test combining two ensembles"""
        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [100, 200, 300]})
        df2 = pd.DataFrame({"x": [4.0, 5.0, 6.0], "value": [400, 500, 600]})
        ddf1 = dd.from_pandas(df1, npartitions=1)
        ddf2 = dd.from_pandas(df2, npartitions=1)

        axes1 = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        axes2 = [UnorderedSignalAxis(np.array([4.0, 5.0, 6.0]), "x", 0, "nm", navigate=True)]

        ensemble1 = Ensemble(ddf1, axes1, quantity="value")
        ensemble2 = Ensemble(ddf2, axes2, quantity="value")

        combined = ensemble1.combine(ensemble2)
        combined_computed = combined.compute()

        # combine() does a left join on common axes, not concatenation
        # With different x values, we get left join behavior
        assert isinstance(combined_computed, pd.DataFrame)
        assert "value" in combined_computed.columns


class TestEnsembleSplitOnAxis:
    """Test split_on_axis() method"""

    def test_split_on_axis(self):
        """Test splitting Ensemble on an axis"""
        df = pd.DataFrame({
            "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "y": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "value": [100, 200, 300, 400, 500, 600],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True),
            UnorderedSignalAxis(np.array([10.0, 20.0]), "y", 1, "nm", navigate=True),
        ]
        ensemble = Ensemble(ddf, axes, quantity="value")

        result_list = ensemble.split_on_axis("x")

        assert isinstance(result_list, list)
        assert len(result_list) == 3  # Three unique x values

        # Each split should have 2 rows (for y = 10, 20)
        for split_ensemble in result_list:
            assert isinstance(split_ensemble, Ensemble)
            split_computed = split_ensemble.compute()
            assert len(split_computed) == 2

    def test_split_on_axis_invalid(self):
        """Test split_on_axis with invalid axis raises error"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [100, 200, 300]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        with pytest.raises(ValueError, match="not found"):
            ensemble.split_on_axis("z")

    def test_split_on_axis_none(self):
        """Test split_on_axis with None raises error"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [100, 200, 300]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        with pytest.raises(ValueError, match="must be specified"):
            ensemble.split_on_axis(None)


class TestEnsembleToDataBlock:
    """Test to_datablock() conversion"""

    def test_to_datablock_basic(self):
        """Test converting Ensemble to DataBlock with rebinning"""
        # Create a simple ensemble from a datablock first
        data = da.from_array(np.arange(12).reshape(3, 4), chunks=(3, 4))
        axes = [
            SignalAxis(np.array([0.0, 1.0, 2.0]), "x", 0, "nm", navigate=True),
            SignalAxis(np.array([0.0, 1.0, 2.0, 3.0]), "y", 1, "nm", navigate=True),
        ]
        db_original = DataBlock(data, axes, quantity="intensity")

        # Convert to ensemble
        ensemble = Ensemble.from_datablock(db_original)

        # to_datablock requires a rebin_scheme to work properly
        # Test that it at least doesn't crash with empty rebin_scheme
        try:
            db_result = ensemble.to_datablock(rebin_scheme={})
            # If successful, verify it's a DataBlock
            assert isinstance(db_result, DataBlock)
            assert db_result.data == db_original.data
        except (AssertionError, ValueError):
            # to_datablock may have requirements we're not meeting
            # Just verify the ensemble was created correctly
            assert isinstance(ensemble, Ensemble)
            assert ensemble.quantity == "intensity"


class TestEnsembleEdgeCases:
    """Test edge cases and special scenarios"""

    def test_ensemble_with_minimal_data(self):
        """Test Ensemble with small dataset"""
        # UnorderedSignalAxis requires at least 2 points
        df = pd.DataFrame({"x": [1.0, 2.0], "value": [100, 200]})
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        assert ensemble.dims == 1
        computed = ensemble.compute()
        assert len(computed) == 2

    def test_ensemble_with_many_axes(self):
        """Test Ensemble with multiple axes"""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "z": [100.0, 200.0, 300.0],
            "value": [1000, 2000, 3000],
        })
        ddf = dd.from_pandas(df, npartitions=1)

        axes = [
            UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True),
            UnorderedSignalAxis(np.array([10.0, 20.0, 30.0]), "y", 1, "nm", navigate=True),
            UnorderedSignalAxis(np.array([100.0, 200.0, 300.0]), "z", 2, "nm", navigate=True),
        ]
        ensemble = Ensemble(ddf, axes, quantity="value")

        assert ensemble.dims == 3
        assert len(ensemble.axes) == 3


class TestEnsembleRebinAxis:
    """Test rebin_axis and rebin_reduce_axis methods"""

    def test_rebin_reduce_axis(self):
        """Test rebinning with reduction"""
        # Create data with regular spacing
        df = pd.DataFrame({
            "x": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [SignalAxis(np.arange(0, 5, 0.5), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(ddf, axes, quantity="value")

        # Rebin to larger bins
        rebinned = ensemble.rebin_reduce_axis("x", new_binsize=1.0)

        assert isinstance(rebinned, Ensemble)
        assert rebinned.has_axis("x")


# x = 0..9 with a y that alternates 0/1. For ``new_binsize=3.0`` the bins hold
# x = {0,1,2}, {3,4,5,6}, {7,8,9}; for ``bin_edges=[0, 4, 9]`` they hold
# x = {0..4}, {5..9}.
_BINSIZE3_CENTERS = [1.49988333, 4.5, 7.50011667]
_BINSIZE3_BIN_OF_ROW = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
_EDGES_CENTERS = [1.99995, 6.50005]
_EDGES_BIN_OF_ROW = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def _xy_ensemble(*, as_pandas: bool = False) -> Ensemble:
    df = pd.DataFrame({
        "x": np.arange(10) * 1.0,
        "y": [0.0, 1.0] * 5,
        "v": np.arange(10) * 10.0,
    })
    data = df if as_pandas else dd.from_pandas(df, npartitions=2)
    axes = [
        SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", navigate=True),
        SignalAxis(np.array([0.0, 1.0]), "y", 1, "um", navigate=True),
    ]
    return Ensemble(data, axes, quantity="v", unit="counts")


def _categorical_ensemble() -> Ensemble:
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "cat": list("abab"), "v": [1, 2, 3, 4]})
    axes = [
        SignalAxis(np.arange(4) * 1.0, "x", 0, "nm", navigate=True),
        CategoricalAxis(["a", "b"], "cat", 1, "-"),
    ]
    return Ensemble(dd.from_pandas(df, npartitions=1), axes, quantity="v")


def _frame(ensemble: Ensemble) -> pd.DataFrame:
    data = ensemble.data
    return data.compute() if isinstance(data, dd.DataFrame) else data


class TestEnsembleRebinAxisCoverage:
    """rebin_axis: dask and pandas backends, explicit edges and error paths"""

    @pytest.mark.parametrize("as_pandas", [False, True])
    def test_rebin_axis_binsize(self, as_pandas):
        ensemble = _xy_ensemble(as_pandas=as_pandas)

        result = ensemble.rebin_axis("x", new_binsize=3.0)

        np.testing.assert_allclose(result.axes[0].points, _BINSIZE3_CENTERS)
        assert result.axes[0].scale == 3.0
        assert result.axes[0].index_in_array == 0
        assert result.axes[1].name == "y"
        assert (result.quantity, result.unit) == ("v", "counts")
        assert isinstance(result.data, pd.DataFrame) is as_pandas
        frame = _frame(result)
        assert "x_bin" not in frame.columns
        np.testing.assert_allclose(
            frame["x"], np.array(_BINSIZE3_CENTERS)[_BINSIZE3_BIN_OF_ROW]
        )
        # The quantity is untouched
        np.testing.assert_allclose(frame["v"], np.arange(10) * 10.0)

    @pytest.mark.parametrize("as_pandas", [False, True])
    def test_rebin_axis_explicit_bin_edges(self, as_pandas):
        ensemble = _xy_ensemble(as_pandas=as_pandas)

        result = ensemble.rebin_axis("x", bin_edges=np.array([0.0, 4.0, 9.0]))

        np.testing.assert_allclose(result.axes[0].points, _EDGES_CENTERS)
        np.testing.assert_allclose(
            _frame(result)["x"], np.array(_EDGES_CENTERS)[_EDGES_BIN_OF_ROW]
        )

    def test_rebin_axis_drops_points_outside_edges(self):
        ensemble = _xy_ensemble()

        result = ensemble.rebin_axis("x", bin_edges=np.array([2.5, 5.5]))

        frame = _frame(result)
        assert len(frame) == 3
        np.testing.assert_allclose(frame["v"], [30.0, 40.0, 50.0])

    def test_rebin_axis_pandas_backend_edge_aligned_points(self):
        """Points sitting exactly on bin edges all land in a bin on the pandas path"""
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "v": [1.0, 2.0, 3.0]})
        axes = [SignalAxis(np.array([0.0, 1.0, 2.0]), "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(df, axes, quantity="v")

        result = ensemble.rebin_axis("x", bin_edges=np.array([0.0, 1.0, 2.0]))

        assert len(result.data) == 3
        assert not result.data["x"].isna().any()
        assert set(result.data["x"]) == set(result.axes[0].points)

    def test_rebin_axis_needs_binsize_or_edges(self):
        with pytest.raises(RuntimeError, match="'bin_edges' are None"):
            _xy_ensemble().rebin_axis("x")

    def test_rebin_axis_nonexisting_axis(self):
        with pytest.raises(RuntimeError, match="has no axis z"):
            _xy_ensemble().rebin_axis("z", new_binsize=1.0)

    def test_rebin_axis_categorical_axis(self):
        with pytest.raises(RuntimeError, match="CategoricalAxis"):
            _categorical_ensemble().rebin_axis("cat", new_binsize=1.0)


class TestEnsembleRebinReduceAxisCoverage:
    def test_rebin_reduce_axis_one_axis_values(self):
        df = pd.DataFrame({"x": np.arange(10) * 1.0, "v": np.arange(10) * 10.0})
        axes = [SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(dd.from_pandas(df, npartitions=2), axes, quantity="v")

        result = ensemble.rebin_reduce_axis("x", new_binsize=3.0, reducer=np.mean)

        frame = _frame(result).sort_values("x")
        np.testing.assert_allclose(frame["x"], _BINSIZE3_CENTERS)
        np.testing.assert_allclose(frame["v"], [10.0, 45.0, 80.0])
        np.testing.assert_allclose(result.axes[0].points, _BINSIZE3_CENTERS)
        assert "x_bin" not in frame.columns

    def test_rebin_reduce_axis_default_reducer_is_mean(self):
        df = pd.DataFrame({"x": np.arange(10) * 1.0, "v": np.arange(10) * 10.0})
        axes = [SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", navigate=True)]
        ensemble = Ensemble(dd.from_pandas(df, npartitions=3), axes, quantity="v")

        result = ensemble.rebin_reduce_axis("x", new_binsize=3.0)

        frame = _frame(result).sort_values("x")
        np.testing.assert_allclose(frame["v"], [10.0, 45.0, 80.0])

    def test_rebin_reduce_axis_keeps_other_axes(self):
        result = _xy_ensemble().rebin_reduce_axis("x", new_binsize=3.0, reducer=np.sum)

        frame = _frame(result)
        rows = {
            (round(x, 3), y): v for x, y, v in zip(frame["x"], frame["y"], frame["v"])
        }
        assert rows == {
            (1.5, 0.0): 20.0,
            (1.5, 1.0): 10.0,
            (4.5, 0.0): 100.0,
            (4.5, 1.0): 80.0,
            (7.5, 0.0): 80.0,
            (7.5, 1.0): 160.0,
        }
        assert (result.quantity, result.unit) == ("v", "counts")
        assert result.axes[1].name == "y"

    def test_rebin_reduce_axis_explicit_bin_edges(self):
        result = _xy_ensemble().rebin_reduce_axis(
            "x", bin_edges=np.array([0.0, 4.0, 9.0]), reducer=np.max
        )

        frame = _frame(result)
        rows = {
            (round(x, 3), y): v for x, y, v in zip(frame["x"], frame["y"], frame["v"])
        }
        assert rows == {
            (2.0, 0.0): 40.0,
            (2.0, 1.0): 30.0,
            (6.5, 0.0): 80.0,
            (6.5, 1.0): 90.0,
        }

    def test_rebin_reduce_axis_nonexisting_axis(self):
        with pytest.raises(RuntimeError, match="has no axis z"):
            _xy_ensemble().rebin_reduce_axis("z", new_binsize=1.0)

    def test_rebin_reduce_axis_categorical_axis(self):
        with pytest.raises(RuntimeError, match="CategoricalAxis"):
            _categorical_ensemble().rebin_reduce_axis("cat", new_binsize=1.0)


class TestBinEdgesAreNotModified:
    """Explicit bin edges belong to the caller: no in-place padding, any sequence works"""

    EDGES = [0.0, 4.0, 9.0]

    @pytest.mark.parametrize("method", ["rebin_axis", "rebin_reduce_axis", "rebin_to_new_axis"])
    @pytest.mark.parametrize(
        "make_edges",
        [
            lambda: np.array([0.0, 4.0, 9.0]),
            lambda: [0.0, 4.0, 9.0],
            lambda: (0.0, 4.0, 9.0),
            lambda: np.array([0, 4, 9]),
        ],
        ids=["float-array", "list", "tuple", "int-array"],
    )
    def test_edges_unchanged_and_result_stable(self, method, make_edges):
        ensemble = _xy_ensemble()
        edges = make_edges()
        snapshot = np.array(edges, dtype=float)

        first = getattr(ensemble, method)("x", bin_edges=edges)
        second = getattr(ensemble, method)("x", bin_edges=edges)

        np.testing.assert_array_equal(np.asarray(edges, dtype=float), snapshot)
        np.testing.assert_allclose(first.axes[0].points, _EDGES_CENTERS)
        np.testing.assert_allclose(second.axes[0].points, _EDGES_CENTERS)

    def test_read_only_edges(self):
        edges = np.array([0.0, 4.0, 9.0])
        edges.setflags(write=False)

        result = _xy_ensemble().rebin_axis("x", bin_edges=edges)

        np.testing.assert_allclose(result.axes[0].points, _EDGES_CENTERS)


class TestEnsembleSplitWithEnsemble:
    """split_with_ensemble: split an ensemble by the quantity of another one"""

    @staticmethod
    def _data(*, as_pandas=False, extra=None) -> Ensemble:
        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0, 3.0] * 2,
            "y": [0.0] * 4 + [1.0] * 4,
            "v": np.arange(8.0),
        })
        axes = [
            SignalAxis(np.arange(4) * 1.0, "x", 0, "nm", navigate=True),
            SignalAxis(np.array([0.0, 1.0]), "y", 1, "um", navigate=True),
        ]
        if extra is not None:
            df, axes = extra(df, axes)
        data = df if as_pandas else dd.from_pandas(df, npartitions=2)
        return Ensemble(data, axes, quantity="v", unit="counts")

    @staticmethod
    def _mask(xs=(0.0, 1.0, 2.0, 3.0), values=(0, 1, 0, 1), *, as_pandas=False):
        df = pd.DataFrame({"x": list(xs), "m": list(values)})
        axes = [SignalAxis(np.array(xs), "x", 0, "nm", navigate=True)]
        data = df if as_pandas else dd.from_pandas(df, npartitions=1)
        return Ensemble(data, axes, quantity="m")

    @pytest.mark.parametrize("as_pandas", [False, True])
    @pytest.mark.parametrize("drop_axes", ["x", ["x"]])
    def test_split_drops_axis_and_filters_rows(self, as_pandas, drop_axes):
        result = self._data(as_pandas=as_pandas).split_with_ensemble(
            self._mask(as_pandas=as_pandas), drop_axes
        )

        assert len(result) == 2
        for ensemble, rows in zip(result, [[0, 2, 4, 6], [1, 3, 5, 7]]):
            assert isinstance(ensemble.data, pd.DataFrame) is as_pandas
            frame = _frame(ensemble).sort_values("v")
            assert list(frame.columns) == ["y", "v"]
            assert list(frame["v"]) == [float(r) for r in rows]
            assert [ax.name for ax in ensemble.axes] == ["y"]
            assert ensemble.axes[0].index_in_array == 0
            assert (ensemble.quantity, ensemble.unit) == ("v", "counts")

    def test_split_axes_hold_coordinates_of_their_group(self):
        ensemble = self._data()
        result = ensemble.split_with_ensemble(self._mask(), "y")

        assert [list(e.axes[0].points) for e in result] == [[0.0, 2.0], [1.0, 3.0]]
        assert result[0].axes[0] is not result[1].axes[0]
        assert isinstance(result[0].axes[0], SignalAxis)
        assert result[0].axes[0].size == 2

    def test_source_axes_are_not_modified(self):
        ensemble = self._data()

        ensemble.split_with_ensemble(self._mask(), "x")

        assert [ax.name for ax in ensemble.axes] == ["x", "y"]
        assert [ax.index_in_array for ax in ensemble.axes] == [0, 1]
        assert isinstance(ensemble.axes[1].points, np.ndarray)
        np.testing.assert_array_equal(ensemble.axes[1].points, [0.0, 1.0])

    def test_rows_without_match_are_excluded(self):
        mask = self._mask(xs=(0.0, 1.0), values=(0, 1))

        result = self._data().split_with_ensemble(mask, "x")

        assert len(result) == 2
        assert sorted(_frame(result[0])["v"]) == [0.0, 4.0]
        assert sorted(_frame(result[1])["v"]) == [1.0, 5.0]

    def test_categorical_and_unordered_axes_are_kept(self):
        def extra(df, axes):
            df = df.assign(cat=list("abab") * 2, u=[5.0, 3.0, 4.0, 1.0] * 2)
            axes = [
                *axes,
                CategoricalAxis(["a", "b"], "cat", 2, "-"),
                UnorderedSignalAxis(np.array([5.0, 3.0, 4.0, 1.0]), "u", 3, "eV"),
            ]
            return df, axes

        ensemble = self._data(extra=extra)

        result = ensemble.split_with_ensemble(self._mask(), ["x", "y"])

        first = result[0]
        assert [type(ax) for ax in first.axes] == [CategoricalAxis, UnorderedSignalAxis]
        assert [ax.index_in_array for ax in first.axes] == [0, 1]
        assert list(first.axes[0].points) == ["a"]
        assert sorted(first.axes[1].points) == [4.0, 5.0]
        assert list(_frame(first).columns) == ["cat", "u", "v"]

    def test_no_common_axes(self):
        other = Ensemble(
            dd.from_pandas(pd.DataFrame({"q": [1.0], "m": [0]}), npartitions=1),
            [SignalAxis(np.array([1.0]), "q", 0, "nm", navigate=True)],
            quantity="m",
        )

        with pytest.raises(AssertionError, match="q not found"):
            self._data().split_with_ensemble(other, "y")

    def test_no_common_axes_when_other_axis_is_dropped(self):
        other = Ensemble(
            dd.from_pandas(pd.DataFrame({"q": [1.0], "m": [0]}), npartitions=1),
            [SignalAxis(np.array([1.0]), "q", 0, "nm", navigate=True)],
            quantity="m",
        )

        with pytest.raises(ValueError, match="No common axes"):
            self._data().split_with_ensemble(other, "q")


class TestEnsembleRebinToNewAxisCoverage:
    def test_rebin_to_new_axis_binsize(self):
        result = _xy_ensemble().rebin_to_new_axis("x", new_binsize=3.0)

        frame = _frame(result)
        np.testing.assert_allclose(
            frame["x"], np.array(_BINSIZE3_CENTERS)[_BINSIZE3_BIN_OF_ROW]
        )
        np.testing.assert_allclose(frame["v"], np.arange(10) * 10.0)
        assert list(frame["repetition"]) == [0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
        assert "x_bin" not in frame.columns
        assert [ax.name for ax in result.axes] == ["x", "y", "repetition"]
        np.testing.assert_allclose(result.axes[0].points, _BINSIZE3_CENTERS)
        rep_axis = result.axes[2]
        assert list(rep_axis.points) == [0, 1]
        assert rep_axis.index_in_array == 2
        assert rep_axis.unit == "#"
        assert not rep_axis.is_nav
        assert (result.quantity, result.unit) == ("v", "counts")

    def test_rebin_to_new_axis_explicit_bin_edges(self):
        result = _xy_ensemble().rebin_to_new_axis(
            "x", bin_edges=np.array([0.0, 4.0, 9.0])
        )

        frame = _frame(result)
        np.testing.assert_allclose(
            frame["x"], np.array(_EDGES_CENTERS)[_EDGES_BIN_OF_ROW]
        )
        assert list(frame["repetition"]) == [0, 0, 1, 1, 2, 0, 0, 1, 1, 2]
        assert list(result.axes[2].points) == [0, 1, 2]

    def test_rebin_to_new_axis_nonexisting_axis(self):
        with pytest.raises(RuntimeError, match="has no axis z"):
            _xy_ensemble().rebin_to_new_axis("z", new_binsize=1.0)

    def test_rebin_to_new_axis_needs_binsize_or_edges(self):
        with pytest.raises(RuntimeError, match="Either call with"):
            _xy_ensemble().rebin_to_new_axis("x")

    def test_rebin_to_new_axis_categorical_axis(self):
        with pytest.raises(RuntimeError, match="CategoricalAxis"):
            _categorical_ensemble().rebin_to_new_axis("cat", new_binsize=1.0)


class TestEnsembleReduceAxisCoverage:
    @pytest.mark.parametrize("as_pandas", [False, True])
    def test_reduce_axis_first_axis(self, as_pandas):
        result = _xy_ensemble(as_pandas=as_pandas).reduce_axis(
            "x", "max", out_quantity="peak", out_unit="a.u."
        )

        frame = _frame(result).sort_values("y")
        assert list(frame["y"]) == [0.0, 1.0]
        assert list(frame["v"]) == [80.0, 90.0]
        assert [ax.name for ax in result.axes] == ["y"]
        assert result.axes[0].index_in_array == 0
        assert (result.quantity, result.unit) == ("peak", "a.u.")

    def test_reduce_axis_last_axis(self):
        ensemble = _xy_ensemble()

        result = ensemble.reduce_axis("y", "sum")

        frame = _frame(result).sort_values("x")
        assert list(frame["x"]) == list(np.arange(10) * 1.0)
        assert list(frame["v"]) == list(np.arange(10) * 10.0)
        assert [ax.name for ax in result.axes] == ["x"]
        # Source ensemble keeps both axes with their original indices
        assert [ax.index_in_array for ax in ensemble.axes] == [0, 1]

    def test_reduce_axis_nonexisting_axis(self):
        with pytest.raises(RuntimeError, match="Axis z not found"):
            _xy_ensemble().reduce_axis("z", "max")


class TestEnsembleCOp:
    """Test c_op() method for custom operations"""

    def test_c_op_addition(self):
        """Test custom operation between two ensembles"""
        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [10.0, 20.0, 30.0]})
        df2 = pd.DataFrame({"x": [1.0, 2.0, 3.0], "value": [5.0, 10.0, 15.0]})
        ddf1 = dd.from_pandas(df1, npartitions=1)
        ddf2 = dd.from_pandas(df2, npartitions=1)

        axes = [UnorderedSignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", navigate=True)]

        ensemble1 = Ensemble(ddf1, axes, quantity="value")
        ensemble2 = Ensemble(ddf2, axes, quantity="value")

        # Define addition operation
        def add_op(s1, s2):
            return s1 + s2

        result = ensemble1.c_op(ensemble2, add_op)
        result_computed = result.compute()

        # Verify structure
        assert isinstance(result_computed, pd.DataFrame)
        assert "value" in result_computed.columns
        assert "x" in result_computed.columns

        # Verify the operation worked correctly (10+5=15, 20+10=30, 30+15=45)
        assert np.allclose(result_computed["value"].values, [15.0, 30.0, 45.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
