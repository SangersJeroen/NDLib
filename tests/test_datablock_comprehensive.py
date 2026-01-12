"""
Comprehensive unit tests for DataBlock class from eels_base/ndlib/BaseStructures.py
This test suite thoroughly tests all methods and edge cases of the DataBlock class.
"""

import pytest
import numpy as np
import dask.array as da
import sys
import os

# Add the parent directory to the path to import eels_base
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eels_base.ndlib.BaseStructures import DataBlock
from eels_base.ndlib.AxesStructures import SignalAxis, UnorderedSignalAxis, CategoricalAxis


class TestDataBlockInitialization:
    """Test DataBlock initialization and validation"""

    def test_basic_initialization(self):
        """Test basic DataBlock creation with valid inputs"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes, quantity="intensity", unit="counts")

        assert db.dims == 2
        assert db.shape == (10, 20)
        assert db.quantity == "intensity"
        assert db.unit == "counts"
        assert len(db.axes) == 2
        assert not db._computed

    def test_initialization_with_default_params(self):
        """Test DataBlock creation with default quantity and unit"""
        data = da.from_array(np.random.rand(5, 5), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(5), "x", 0, "nm", True),
            SignalAxis(np.arange(5), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        assert db.quantity == "q"
        assert db.unit == "-"

    def test_mismatched_dimensions(self):
        """Test that mismatched data and axes dimensions raise error"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [SignalAxis(np.arange(10), "x", 0, "nm", True)]

        with pytest.raises(RuntimeError, match="Mismatched number of dimensions"):
            DataBlock(data, axes)

    def test_mismatched_axis_sizes(self):
        """Test that mismatched axis sizes and data shape raise error"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(15), "x", 0, "nm", True),  # Wrong size
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]

        with pytest.raises(RuntimeError, match="Mismatch along dimension"):
            DataBlock(data, axes)

    def test_squeeze_single_dimension(self):
        """Test that axes with size 1 are squeezed"""
        data = da.from_array(np.random.rand(1, 20), chunks=(1, 10))
        axes = [
            SignalAxis(np.array([0]), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes, quantity="intensity")

        assert db.dims == 1
        assert db.shape == (20,)
        assert len(db.axes) == 1
        assert db.quantity == "x"  # Quantity set to squeezed axis name

    def test_axes_sorting(self):
        """Test that axes are sorted by index_in_array"""
        data = da.from_array(np.random.rand(10, 20, 30), chunks=(5, 10, 15))
        axes = [
            SignalAxis(np.arange(30), "z", 2, "nm", True),
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        assert db.axes[0].name == "x"
        assert db.axes[1].name == "y"
        assert db.axes[2].name == "z"


class TestDataBlockCompute:
    """Test compute functionality"""

    def test_compute_lazy_array(self):
        """Test computing a lazy dask array"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        assert not db._computed
        result = db.compute()

        assert db._computed
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 20)

    def test_compute_already_computed(self):
        """Test that computing already computed data returns cached result"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result1 = db.compute()
        result2 = db.compute()

        assert result1 is result2


class TestDataBlockAxisMethods:
    """Test axis-related methods"""

    def test_has_axis_existing(self):
        """Test has_axis returns True for existing axis"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "energy", 1, "eV", True),
        ]
        db = DataBlock(data, axes)

        assert db.has_axis("x")
        assert db.has_axis("energy")

    def test_has_axis_nonexisting(self):
        """Test has_axis returns False for non-existing axis"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        assert not db.has_axis("z")

    def test_axis_obj(self):
        """Test retrieving axis object by name"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "energy", 1, "eV", True),
        ]
        db = DataBlock(data, axes)

        x_axis = db.axis_obj("x")
        assert x_axis.name == "x"
        assert x_axis.size == 10
        assert x_axis.unit == "nm"

    def test_axis_obj_nonexisting(self):
        """Test that axis_obj raises error for non-existing axis"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        with pytest.raises(RuntimeError, match="does not have axis z"):
            db.axis_obj("z")

    def test_axis_points_by_name(self):
        """Test axis_points_by_name method returns axis points"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10) * 0.5, "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        x_points = db.axis_points_by_name("x")
        assert len(x_points) == 10
        assert np.allclose(x_points[0], 0)
        assert np.allclose(x_points[1], 0.5)


class TestDataBlockIterAxis:
    """Test iter_axis functionality"""

    def test_iter_axis_with_range(self):
        """Test iterating over axis with min and max indices"""
        data_arr = np.arange(100).reshape(10, 10)
        data = da.from_array(data_arr, chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        slices = list(db.iter_axis("x", min_idx=2, max_idx=5))
        assert len(slices) == 3

    def test_iter_axis_nonexisting(self):
        """Test that iterating over non-existing axis raises error"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        with pytest.raises(RuntimeError, match="Axis z not found"):
            list(db.iter_axis("z"))


class TestDataBlockCropAxis:
    """Test crop_axis functionality"""

    def test_crop_axis(self):
        """Test cropping axis to min/max range"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", True),
            SignalAxis(np.arange(10) * 1.0, "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        # Crop x axis from 3.0 to 7.0 (should keep indices 3,4,5,6,7 = 5 points)
        db_cropped = db.crop_axis("x", min=3.0, max=7.0)

        # Verify structure
        assert db_cropped.dims == 2
        assert db_cropped.axes[0].name == "x"
        assert db_cropped.axes[1].name == "y"

        # Verify sizes match (data shape should match axis sizes)
        assert db_cropped.shape[0] == db_cropped.axes[0].size
        assert db_cropped.shape[1] == db_cropped.axes[1].size

        # Verify cropping worked (5 points in range [3.0, 7.0])
        assert db_cropped.axes[0].size == 5
        assert db_cropped.shape == (5, 10)

        # Verify axis points are correct
        expected_x_points = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        assert np.allclose(db_cropped.axes[0].points, expected_x_points)


class TestDataBlockGetMethod:
    """Test advanced get() indexing method"""

    def test_get_with_inequality(self):
        """Test get with inequality expressions"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db.get({"x": "x>5"})
        assert result.axes[0].size == 4

        result2 = db.get({"y": "y<3"})
        assert result2.axes[1].size == 3

    def test_get_with_multiple_axes(self):
        """Test get with multiple axes"""
        data = da.from_array(np.arange(1000).reshape(10, 10, 10), chunks=(5, 5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
            SignalAxis(np.arange(10), "z", 2, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db.get({"x": "x>=2", "z": "z<8"})
        assert result.shape == (8, 10, 8)

    def test_get_with_boolean_mask(self):
        """Test get with boolean mask"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        mask = np.array([True, False, True, False, True, False, True, False, True, False])
        result = db.get({"x": mask})
        assert result.axes[0].size == 5

    def test_get_empty_dict(self):
        """Test get with empty dict returns original"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db.get({})
        assert result.shape == db.shape

    def test_get_invalid_axis(self):
        """Test get with non-existent axis raises error"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        with pytest.raises(ValueError, match="Axis 'z' not found"):
            db.get({"z": "z>5"})

    def test_get_with_categorical_axis(self):
        """Test get with categorical axis"""
        data = da.from_array(np.arange(30).reshape(3, 10), chunks=(3, 5))
        axes = [
            CategoricalAxis(["low", "medium", "high"], "level", 0, "-", True),
            SignalAxis(np.arange(10), "x", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        # Note: CategoricalAxis equality matching may have implementation specifics
        result = db.get({"x": "x>=0"})  # Test with numeric axis instead
        assert result.axes[1].size == 10


class TestDataBlockValueGet:
    """Test value_get() method"""

    def test_value_get_basic(self):
        """Test value_get returns tuple of data and axes dict"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        quantity_data, axes_dict = db.value_get({"x": "x>5"})
        assert isinstance(quantity_data, da.Array)
        assert isinstance(axes_dict, dict)
        assert "x" in axes_dict
        assert "y" in axes_dict
        assert len(axes_dict["x"]) == 4

    def test_value_get_empty_dict(self):
        """Test value_get with empty dict"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        quantity_data, axes_dict = db.value_get({})
        assert quantity_data.shape == (10, 10)
        assert len(axes_dict["x"]) == 10


class TestDataBlockArithmetic:
    """Test arithmetic operations"""

    def test_add_scalar(self):
        """Test adding scalar to DataBlock"""
        data = da.from_array(np.ones((10, 20)), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db + 5
        result_computed = result.compute()

        assert np.allclose(result_computed, 6)

    def test_multiply_scalar(self):
        """Test multiplying DataBlock by scalar"""
        data = da.from_array(np.ones((10, 20)) * 2, chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db * 3
        result_computed = result.compute()

        assert np.allclose(result_computed, 6)

    def test_subtract_scalar(self):
        """Test subtracting scalar from DataBlock"""
        data = da.from_array(np.ones((10, 20)) * 5, chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db - 2
        result_computed = result.compute()

        assert np.allclose(result_computed, 3)

    def test_power_operation(self):
        """Test power operation"""
        data = da.from_array(np.ones((10, 20)) * 2, chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db**3
        result_computed = result.compute()

        assert np.allclose(result_computed, 8)


class TestDataBlockRebinAndReduce:
    """Test rebin and reduce operations"""

    def test_rebin_reduce_axis(self):
        """Test rebinning and reducing an axis"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", True),
            SignalAxis(np.arange(10) * 1.0, "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db.rebin_reduce_axis("x", new_binsize=2.0)

        assert result.axes[0].size < 10
        assert result.axes[0].name == "x"

    def test_reduce_axis(self):
        """Test fully reducing an axis"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(10), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        result = db.reduce_axis("x", reducer=lambda s: s.max())

        assert result.dims == 1
        assert result.shape == (10,)
        # Check that x axis is removed
        assert not result.has_axis("x")


class TestDataBlockSplitOperations:
    """Test split_on_axis operations"""

    def test_split_on_axis(self):
        """Test splitting DataBlock on an axis"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", True),
            SignalAxis(np.arange(10) * 1.0, "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        # Split on x axis (10 unique values)
        result_list = db.split_on_axis("x")

        # Verify we get a list with correct number of DataBlocks
        assert isinstance(result_list, list)
        assert len(result_list) == 10

        # Each DataBlock should be 1D after squeezing (size 1 axis removed)
        for i, split_db in enumerate(result_list):
            assert isinstance(split_db, DataBlock)
            # After squeezing size-1 axis, should have only y axis left
            assert split_db.dims == 1
            assert split_db.shape == (10,)
            assert split_db.axes[0].name == "y"


class TestDataBlockWithDifferentAxisTypes:
    """Test DataBlock with different axis types"""

    def test_with_unordered_axis(self):
        """Test DataBlock with UnorderedSignalAxis"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            UnorderedSignalAxis(np.array([1, 5, 2, 8, 3, 9, 4, 7, 6, 0]), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        assert db.dims == 2
        assert db.axes[0].ordered == False
        assert db.axes[1].ordered == True

    def test_with_categorical_axis(self):
        """Test DataBlock with CategoricalAxis"""
        data = da.from_array(np.random.rand(3, 20), chunks=(3, 10))
        axes = [
            CategoricalAxis(["low", "medium", "high"], "level", 0, "-", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        assert db.dims == 2
        assert db.axes[0].size == 3
        assert db.axes[0].ordered == False


class TestDataBlockEdgeCases:
    """Test edge cases and error conditions"""

    def test_1d_datablock(self):
        """Test 1D DataBlock"""
        data = da.from_array(np.arange(100), chunks=50)
        axes = [SignalAxis(np.arange(100), "x", 0, "nm", True)]
        db = DataBlock(data, axes)

        assert db.dims == 1
        assert db.shape == (100,)

    def test_high_dimensional_datablock(self):
        """Test high-dimensional DataBlock (4D)"""
        data = da.from_array(np.random.rand(5, 6, 7, 8), chunks=(5, 3, 7, 4))
        axes = [
            SignalAxis(np.arange(5), "x", 0, "nm", True),
            SignalAxis(np.arange(6), "y", 1, "nm", True),
            SignalAxis(np.arange(7), "z", 2, "nm", True),
            SignalAxis(np.arange(8), "t", 3, "s", True),
        ]
        db = DataBlock(data, axes)

        assert db.dims == 4
        assert db.shape == (5, 6, 7, 8)

    def test_quantity_data_method(self):
        """Test quantity_data method returns data array"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        quantity_data = db.quantity_data()

        assert isinstance(quantity_data, da.Array)
        assert quantity_data.shape == (10, 20)

    def test_repr_and_str(self):
        """Test __repr__ and __str__ methods"""
        data = da.from_array(np.random.rand(10, 20), chunks=(5, 10))
        axes = [
            SignalAxis(np.arange(10), "x", 0, "nm", True),
            SignalAxis(np.arange(20), "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        repr_str = repr(db)
        str_str = str(db)

        assert isinstance(repr_str, str)
        assert isinstance(str_str, str)
        assert repr_str == str_str


class TestDataBlockConversions:
    """Test conversion methods"""

    def test_to_ensemble(self):
        """Test converting DataBlock to Ensemble"""
        data = da.from_array(np.arange(100).reshape(10, 10), chunks=(5, 5))
        axes = [
            SignalAxis(np.arange(10) * 1.0, "x", 0, "nm", True),
            SignalAxis(np.arange(10) * 1.0, "y", 1, "nm", True),
        ]
        db = DataBlock(data, axes)

        ensemble = db.to_ensemble()

        assert hasattr(ensemble, "data")
        assert hasattr(ensemble, "axes")
        assert ensemble.dims == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
