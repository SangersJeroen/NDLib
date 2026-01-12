"""
Comprehensive tests for DataBlock and Ensemble serialization methods.
Tests save/load functionality including lazy loading support.
"""

import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import sys
import os
from pathlib import Path
import shutil

# Add the parent directory to the path to import eels_base
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NDLib.BaseStructures import DataBlock, Ensemble
from NDLib.AxesStructures import SignalAxis, UnorderedSignalAxis, CategoricalAxis


class TestDataBlockSerialization:
    """Test DataBlock save/load functionality"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files"""
        return tmp_path / "test_datablock_serialization"

    @pytest.fixture
    def simple_datablock(self):
        """Create a simple DataBlock for testing"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        return DataBlock(data, axes, quantity="intensity", unit="counts")

    @pytest.fixture
    def complex_datablock(self):
        """Create a more complex DataBlock with different axis types"""
        data = da.from_array(np.random.rand(5, 10, 15), chunks=(5, 5, 10))
        axes = [
            SignalAxis(np.linspace(0, 4, 5), "x", 0, "um", True),
            UnorderedSignalAxis(
                np.array([1.5, 2.3, 3.1, 4.5, 5.0, 6.2, 7.1, 8.4, 9.0, 10.5]), "y", 1, "nm", False
            ),
            SignalAxis(np.arange(15) * 0.5, "energy", 2, "eV", False),
        ]
        return DataBlock(data, axes, quantity="signal", unit="au")

    @pytest.fixture
    def categorical_datablock(self):
        """Create a DataBlock with categorical axis"""
        data = da.from_array(np.random.rand(3, 10), chunks=(3, 5))
        axes = [
            CategoricalAxis(["alpha", "beta", "gamma"], "category", 0, "-", False),
            SignalAxis(np.arange(10), "x", 1, "nm", True),
        ]
        return DataBlock(data, axes, quantity="value", unit="V")

    def test_save_creates_zarr_directory(self, simple_datablock, temp_dir):
        """Test that save creates a .zarr directory"""
        filepath = temp_dir / "test_datablock.zarr"
        simple_datablock.save(filepath)

        assert filepath.exists()
        assert filepath.is_dir()
        assert (filepath / "data").exists()

    def test_save_adds_zarr_extension(self, simple_datablock, temp_dir):
        """Test that save adds .zarr extension if not present"""
        filepath = temp_dir / "test_datablock"
        simple_datablock.save(filepath)

        expected_path = temp_dir / "test_datablock.zarr"
        assert expected_path.exists()

    def test_load_simple_datablock_lazy(self, simple_datablock, temp_dir):
        """Test loading a simple DataBlock with lazy loading"""
        filepath = temp_dir / "test_simple.zarr"
        simple_datablock.save(filepath)

        loaded_db = DataBlock.load(filepath, lazy=True)

        assert loaded_db.quantity == simple_datablock.quantity
        assert loaded_db.unit == simple_datablock.unit
        assert loaded_db.dims == simple_datablock.dims
        assert loaded_db.shape == simple_datablock.shape
        assert len(loaded_db.axes) == len(simple_datablock.axes)

        # Check that data is lazy
        assert isinstance(loaded_db.data, da.Array)

        # Compare actual data values
        np.testing.assert_array_almost_equal(
            loaded_db.data.compute(), simple_datablock.data.compute()
        )

    def test_load_simple_datablock_eager(self, simple_datablock, temp_dir):
        """Test loading a simple DataBlock without lazy loading"""
        filepath = temp_dir / "test_eager.zarr"
        simple_datablock.save(filepath)

        loaded_db = DataBlock.load(filepath, lazy=False)

        assert loaded_db.quantity == simple_datablock.quantity
        assert isinstance(loaded_db.data, da.Array)

        # Compare actual data values
        np.testing.assert_array_almost_equal(
            loaded_db.data.compute(), simple_datablock.data.compute()
        )

    def test_axes_preserved(self, simple_datablock, temp_dir):
        """Test that axes are correctly preserved through save/load"""
        filepath = temp_dir / "test_axes.zarr"
        simple_datablock.save(filepath)

        loaded_db = DataBlock.load(filepath)

        for orig_axis, loaded_axis in zip(simple_datablock.axes, loaded_db.axes):
            assert orig_axis.name == loaded_axis.name
            assert orig_axis.index_in_array == loaded_axis.index_in_array
            assert orig_axis.unit == loaded_axis.unit
            assert orig_axis.is_nav == loaded_axis.is_nav
            assert orig_axis.size == loaded_axis.size
            np.testing.assert_array_almost_equal(orig_axis.points, loaded_axis.points)

    def test_complex_datablock_serialization(self, complex_datablock, temp_dir):
        """Test serialization of DataBlock with mixed axis types"""
        filepath = temp_dir / "test_complex.zarr"
        complex_datablock.save(filepath)

        loaded_db = DataBlock.load(filepath)

        assert loaded_db.dims == complex_datablock.dims
        assert loaded_db.quantity == complex_datablock.quantity

        # Check each axis type
        assert isinstance(loaded_db.axes[0], SignalAxis)
        assert isinstance(loaded_db.axes[1], UnorderedSignalAxis)
        assert isinstance(loaded_db.axes[2], SignalAxis)

        # Compare data
        np.testing.assert_array_almost_equal(
            loaded_db.data.compute(), complex_datablock.data.compute()
        )

    def test_categorical_axis_serialization(self, categorical_datablock, temp_dir):
        """Test serialization of DataBlock with categorical axis"""
        filepath = temp_dir / "test_categorical.zarr"
        categorical_datablock.save(filepath)

        loaded_db = DataBlock.load(filepath)

        assert isinstance(loaded_db.axes[0], CategoricalAxis)
        assert loaded_db.axes[0].name == "category"
        assert list(loaded_db.axes[0].points) == ["alpha", "beta", "gamma"]

        np.testing.assert_array_almost_equal(
            loaded_db.data.compute(), categorical_datablock.data.compute()
        )

    def test_large_datablock_lazy_loading(self, temp_dir):
        """Test that lazy loading doesn't load all data into memory"""
        # Create a larger DataBlock
        data = da.random.random((100, 200, 50), chunks=(20, 40, 25))
        axes = [
            SignalAxis(np.arange(100), "x", 0, "nm", True),
            SignalAxis(np.arange(200), "y", 1, "nm", True),
            SignalAxis(np.arange(50), "z", 2, "nm", True),
        ]
        db = DataBlock(data, axes, quantity="data", unit="a.u.")

        filepath = temp_dir / "test_large.zarr"
        db.save(filepath)

        # Load lazily
        loaded_db = DataBlock.load(filepath, lazy=True)

        # Should be a dask array
        assert isinstance(loaded_db.data, da.Array)

        # Compute a small slice to verify
        slice_orig = db.data[0:10, 0:10, 0:10].compute()
        slice_loaded = loaded_db.data[0:10, 0:10, 0:10].compute()
        np.testing.assert_array_almost_equal(slice_orig, slice_loaded)

    def test_overwrite_existing_file(self, simple_datablock, temp_dir):
        """Test that saving overwrites existing file"""
        filepath = temp_dir / "test_overwrite.zarr"
        simple_datablock.save(filepath)

        # Create different datablock
        new_data = da.from_array(np.ones((10, 20)), chunks=(5, 10))
        new_axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        new_db = DataBlock(new_data, new_axes, quantity="new", unit="new_unit")

        # Save over the old file
        new_db.save(filepath)

        # Load and verify it's the new data
        loaded_db = DataBlock.load(filepath)
        assert loaded_db.quantity == "new"
        assert loaded_db.unit == "new_unit"
        np.testing.assert_array_almost_equal(loaded_db.data.compute(), np.ones((10, 20)))

    def test_metadata_preserved(self, simple_datablock, temp_dir):
        """Test that all metadata is preserved"""
        filepath = temp_dir / "test_metadata.zarr"
        simple_datablock.save(filepath)

        loaded_db = DataBlock.load(filepath)

        assert loaded_db.quantity == simple_datablock.quantity
        assert loaded_db.unit == simple_datablock.unit
        assert loaded_db.dims == simple_datablock.dims
        assert loaded_db.shape == simple_datablock.shape


class TestEnsembleSerialization:
    """Test Ensemble save/load functionality"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files"""
        return tmp_path / "test_ensemble_serialization"

    @pytest.fixture
    def simple_ensemble(self):
        """Create a simple Ensemble for testing"""
        # Create sample data
        x_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        y_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        intensity = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

        df = pd.DataFrame(
            {
                "x": x_vals,
                "y": y_vals,
                "intensity": intensity,
            }
        )
        ddf = dd.from_pandas(df, npartitions=2)

        axes = [
            SignalAxis(np.unique(x_vals), "x", 0, "nm", True),
            SignalAxis(np.unique(y_vals), "y", 1, "nm", True),
        ]

        return Ensemble(ddf, axes, quantity="intensity", unit="counts")

    @pytest.fixture
    def complex_ensemble(self):
        """Create a more complex Ensemble"""
        # Create irregular data
        np.random.seed(42)
        n_points = 50

        x_vals = np.random.uniform(0, 10, n_points)
        y_vals = np.random.uniform(0, 10, n_points)
        signal = np.random.rand(n_points) * 100

        df = pd.DataFrame(
            {
                "x": x_vals,
                "y": y_vals,
                "signal": signal,
            }
        )
        ddf = dd.from_pandas(df, npartitions=4)

        axes = [
            UnorderedSignalAxis(x_vals, "x", 0, "um", True),
            UnorderedSignalAxis(y_vals, "y", 1, "um", False),
        ]

        return Ensemble(ddf, axes, quantity="signal", unit="a.u.")

    def test_save_creates_zarr_directory(self, simple_ensemble, temp_dir):
        """Test that save creates a .zarr directory with parquet file"""
        filepath = temp_dir / "test_ensemble.zarr"
        simple_ensemble.save(filepath)

        assert filepath.exists()
        assert filepath.is_dir()
        assert (filepath / "dataframe.parquet").exists()

    def test_save_adds_zarr_extension(self, simple_ensemble, temp_dir):
        """Test that save adds .zarr extension if not present"""
        filepath = temp_dir / "test_ensemble"
        simple_ensemble.save(filepath)

        expected_path = temp_dir / "test_ensemble.zarr"
        assert expected_path.exists()

    def test_load_simple_ensemble_lazy(self, simple_ensemble, temp_dir):
        """Test loading a simple Ensemble with lazy loading"""
        filepath = temp_dir / "test_simple.zarr"
        simple_ensemble.save(filepath)

        loaded_ens = Ensemble.load(filepath, lazy=True)

        assert loaded_ens.quantity == simple_ensemble.quantity
        assert loaded_ens.unit == simple_ensemble.unit
        assert loaded_ens.dims == simple_ensemble.dims
        assert len(loaded_ens.axes) == len(simple_ensemble.axes)

        # Check that data is lazy
        assert isinstance(loaded_ens.data, dd.DataFrame)

        # Compare actual data values
        pd.testing.assert_frame_equal(
            loaded_ens.data.compute().sort_values(by=["x", "y"]).reset_index(drop=True),
            simple_ensemble.data.compute().sort_values(by=["x", "y"]).reset_index(drop=True),
        )

    def test_load_simple_ensemble_eager(self, simple_ensemble, temp_dir):
        """Test loading a simple Ensemble without lazy loading"""
        filepath = temp_dir / "test_eager.zarr"
        simple_ensemble.save(filepath)

        loaded_ens = Ensemble.load(filepath, lazy=False)

        assert loaded_ens.quantity == simple_ensemble.quantity
        assert isinstance(loaded_ens.data, dd.DataFrame)

        # Compare actual data values
        pd.testing.assert_frame_equal(
            loaded_ens.data.compute().sort_values(by=["x", "y"]).reset_index(drop=True),
            simple_ensemble.data.compute().sort_values(by=["x", "y"]).reset_index(drop=True),
        )

    def test_axes_preserved(self, simple_ensemble, temp_dir):
        """Test that axes are correctly preserved through save/load"""
        filepath = temp_dir / "test_axes.zarr"
        simple_ensemble.save(filepath)

        loaded_ens = Ensemble.load(filepath)

        for orig_axis, loaded_axis in zip(simple_ensemble.axes, loaded_ens.axes):
            assert orig_axis.name == loaded_axis.name
            assert orig_axis.index_in_array == loaded_axis.index_in_array
            assert orig_axis.unit == loaded_axis.unit
            assert orig_axis.is_nav == loaded_axis.is_nav
            assert type(orig_axis).__name__ == type(loaded_axis).__name__
            np.testing.assert_array_almost_equal(orig_axis.points, loaded_axis.points)

    def test_complex_ensemble_serialization(self, complex_ensemble, temp_dir):
        """Test serialization of Ensemble with unordered axes"""
        filepath = temp_dir / "test_complex.zarr"
        complex_ensemble.save(filepath)

        loaded_ens = Ensemble.load(filepath)

        assert loaded_ens.dims == complex_ensemble.dims
        assert loaded_ens.quantity == complex_ensemble.quantity

        # Check axis types
        assert isinstance(loaded_ens.axes[0], UnorderedSignalAxis)
        assert isinstance(loaded_ens.axes[1], UnorderedSignalAxis)

        # Compare data
        orig_df = complex_ensemble.data.compute().sort_values(by=["x", "y"]).reset_index(drop=True)
        loaded_df = loaded_ens.data.compute().sort_values(by=["x", "y"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(orig_df, loaded_df)

    def test_large_ensemble_lazy_loading(self, temp_dir):
        """Test that lazy loading works for larger Ensemble"""
        # Create a larger Ensemble
        np.random.seed(123)
        n_points = 10000

        x_vals = np.random.uniform(0, 100, n_points)
        y_vals = np.random.uniform(0, 100, n_points)
        data_vals = np.random.rand(n_points)

        df = pd.DataFrame(
            {
                "x": x_vals,
                "y": y_vals,
                "data": data_vals,
            }
        )
        ddf = dd.from_pandas(df, npartitions=10)

        axes = [
            UnorderedSignalAxis(x_vals, "x", 0, "nm", True),
            UnorderedSignalAxis(y_vals, "y", 1, "nm", False),
        ]

        ens = Ensemble(ddf, axes, quantity="data", unit="a.u.")

        filepath = temp_dir / "test_large.zarr"
        ens.save(filepath)

        # Load lazily
        loaded_ens = Ensemble.load(filepath, lazy=True)

        # Should be a dask dataframe
        assert isinstance(loaded_ens.data, dd.DataFrame)

        # Verify a subset
        orig_subset = ens.data.head(10).sort_values(by=["x", "y"]).reset_index(drop=True)
        loaded_subset = loaded_ens.data.head(10).sort_values(by=["x", "y"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(orig_subset, loaded_subset)

    def test_overwrite_existing_file(self, simple_ensemble, temp_dir):
        """Test that saving overwrites existing file"""
        filepath = temp_dir / "test_overwrite.zarr"
        simple_ensemble.save(filepath)

        # Create different ensemble
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "new_qty": [7, 8, 9],
            }
        )
        ddf = dd.from_pandas(df, npartitions=1)
        axes = [
            SignalAxis(np.array([1, 2, 3]), "a", 0, "-", True),
            SignalAxis(np.array([4, 5, 6]), "b", 1, "-", False),
        ]
        new_ens = Ensemble(ddf, axes, quantity="new_qty", unit="new_unit")

        # Save over the old file
        new_ens.save(filepath)

        # Load and verify it's the new data
        loaded_ens = Ensemble.load(filepath)
        assert loaded_ens.quantity == "new_qty"
        assert loaded_ens.unit == "new_unit"

    def test_metadata_preserved(self, simple_ensemble, temp_dir):
        """Test that all metadata is preserved"""
        filepath = temp_dir / "test_metadata.zarr"
        simple_ensemble.save(filepath)

        loaded_ens = Ensemble.load(filepath)

        assert loaded_ens.quantity == simple_ensemble.quantity
        assert loaded_ens.unit == simple_ensemble.unit
        assert loaded_ens.dims == simple_ensemble.dims


class TestRoundTripConsistency:
    """Test that multiple save/load cycles preserve data integrity"""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test files"""
        return tmp_path / "test_roundtrip"

    def test_datablock_multiple_roundtrips(self, temp_dir):
        """Test that DataBlock survives multiple save/load cycles"""
        # Create initial DataBlock
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes, quantity="test", unit="units")

        # Save and load 3 times
        for i in range(3):
            filepath = temp_dir / f"roundtrip_{i}.zarr"
            db.save(filepath)
            db = DataBlock.load(filepath, lazy=True)

        # Verify final data matches original
        assert db.quantity == "test"
        assert db.unit == "units"
        assert db.dims == 2
        assert db.shape == (10, 20)

    def test_ensemble_multiple_roundtrips(self, temp_dir):
        """Test that Ensemble survives multiple save/load cycles"""
        # Create initial Ensemble
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "val": [7.0, 8.0, 9.0],
            }
        )
        ddf = dd.from_pandas(df, npartitions=1)
        axes = [
            SignalAxis(np.array([1.0, 2.0, 3.0]), "x", 0, "nm", True),
            SignalAxis(np.array([4.0, 5.0, 6.0]), "y", 1, "nm", True),
        ]
        ens = Ensemble(ddf, axes, quantity="val", unit="units")

        # Save and load 3 times
        for i in range(3):
            filepath = temp_dir / f"roundtrip_{i}.zarr"
            ens.save(filepath)
            ens = Ensemble.load(filepath, lazy=True)

        # Verify final data matches original
        assert ens.quantity == "val"
        assert ens.unit == "units"
        assert ens.dims == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
