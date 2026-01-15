# pyright: basic
from typing import Callable, Literal, Optional, Self, Sequence, TypeAlias, Iterable, Generator
from copy import deepcopy
from pathlib import Path
import json

from .AxesStructures import SignalAxis, UnorderedSignalAxis, CategoricalAxis
from .Types import Number, Axis1D
from .IndexingUtils import (
    parse_inequality_to_mask,
    parse_inequality_to_query,
    parse_categorical_to_mask,
    parse_categorical_to_query,
)

import numpy as np
import dask.array as da
import dask.dataframe as dd
import zarr

import pandas as pd
import sparse

AxisLike: TypeAlias = SignalAxis | UnorderedSignalAxis | CategoricalAxis
TwoSeriesOperation: TypeAlias = Callable[[pd.Series, pd.Series], pd.Series]


def return_axis_idx(axis: SignalAxis | UnorderedSignalAxis) -> int:
    return axis.index_in_array


class DataBlock:
    """Datastructure for n-dimensional structured dataset with coordinate axes.

    Attributes:
        data: dask.array.Array; n-dimensional lazy array containing datapoints
        axes: Sequence[AxisLike]; the axes corresponding to the
            dimensions in data
        dims: int; number of dimensions
        quantity: str; name of the quantity that is stored in data
        unit: str; unit corresponding to the quantity stored in data
        shape: tuple[int, ...]; shape of data
    """

    def __init__(
        self,
        data: da.Array,
        axes: Iterable[SignalAxis],
        quantity: str = "q",
        unit: str = "-",
    ):
        axes_list: list = list(axes)
        if not len(data.shape) == len(axes_list):
            raise RuntimeError("Mismatched number of dimensions in data and axes")
        _shape = data.shape

        axes = sorted(axes, key=return_axis_idx)

        for i in range(len(axes)):
            axis = axes[i]
            if _shape[i] != len(axis.points):
                raise RuntimeError(
                    f"Shape along dimension {i} has length {_shape[i]} but axis has length {len(axis.points)}"
                )
            if _shape[i] != axis.size:
                raise RuntimeError(
                    f"Shape along dimension {i} has length {_shape[i]} but axis has size {axis.size}"
                )

        # If any axis has size 1, we squeezing the data and set the quantity to the axis name
        if any(axis.size == 1 for axis in axes_list):
            print("DataBlock: Squeezing data along size 1 axes")
            squeeze_axes = [axis.index_in_array for axis in axes_list if axis.size == 1]
            data = da.squeeze(data, axis=tuple(squeeze_axes))
            for axis in axes_list:
                if axis.size == 1:
                    quantity = axis.name
            axes_list = [axis for axis in axes_list if axis.size > 1]

        self.data: da.Array = data
        self.axes: list[SignalAxis] = axes_list
        self.dims: int = len(self.data.shape)
        self.quantity: str = quantity
        self.unit: str = unit
        if isinstance(self.data, da.Array):
            self._computed: bool = False
        else:
            self._computed = True
        self.__post_init_db__()

    def __repr__(self) -> str:
        return f"{self.__class__}\n"

    def __str__(self) -> str:
        return self.__repr__()

    def __post_init_db__(self) -> None:
        self.axes.sort(key=return_axis_idx)
        axis_sizes = [axis.size for axis in self.axes]
        data_shape = self.data.shape
        for dim in range(self.dims):
            if not axis_sizes[dim] == data_shape[dim]:
                raise RuntimeError(
                    f"Mismatch along dimension {dim}\n{self.axes[dim].name} has length {axis_sizes[dim]} but data has length {data_shape[dim]} along dimension"
                )
        self.shape: tuple[int, ...] = tuple([axis.size for axis in self.axes])

    def reorder_axis(self, axis_name: str, target_index: int) -> None:
        """
        Allows reording of both axis and data.
        Will call `__post_init_db__` to verify data shape and axes match.

        Args:
            axis_name: str, name of axis to reorder 
            target_index: int, dimensional position to move axis to

        Raises:
            RuntimeError: If axis with `axis_name` not known.
        """
        if not self.has_axis(axis_name):
            raise RuntimeError(f'{self} has no axis: {axis_name}')

        old_axes = deepcopy(self.axes)
        old_index = self.axis_obj(axis_name).index_in_array
        if not self.axes[target_index].name == axis_name:
            axis_at_target = old_axes[target_index]

            old_axes[target_index] = self.axes[old_index]
            old_axes[target_index].index_in_array = target_index
            old_axes[old_index] = axis_at_target
            old_axes[old_index].index_in_array = old_index

            self.axes = old_axes
            self.data = da.swapaxes(self.data, target_index, old_index)

            self.__post_init_db__()
        else:
            print(f'axis {axis_name} already at index {target_index}')

    def compute(self) -> np.ndarray:
        """Computes or returns the quantity in self.data depending on
        whether it has been computed before.

        Returns:
            np.ndarray; computed data
        """
        if not self._computed:
            self.data = self.data.compute()
            self._computed = True
        return self.data

    def save(self, filepath: str | Path) -> None:
        """Save DataBlock to disk using Zarr format for lazy loading.

        Args:
            filepath: str or Path; path where the DataBlock will be saved
        """
        filepath = Path(filepath)
        if filepath.suffix != ".zarr":
            filepath = filepath.with_suffix(".zarr")

        # Save data array to zarr
        data_path = str(filepath / "data")
        if isinstance(self.data, da.Array):
            self.data.to_zarr(data_path, overwrite=True)
        else:
            zarr.save_array(data_path, self.data, overwrite=True)

        # Save axes information
        axes_data = []
        for axis in self.axes:
            axis_dict = {
                "type": type(axis).__name__,
                "name": axis.name,
                "index_in_array": axis.index_in_array,
                "unit": axis.unit,
                "navigate": axis.is_nav,
            }
            if isinstance(axis, (SignalAxis, UnorderedSignalAxis)):
                axis_dict["points"] = (
                    axis.points.tolist() if hasattr(axis.points, "tolist") else list(axis.points)
                )
            elif isinstance(axis, CategoricalAxis):
                axis_dict["categories"] = axis.points.tolist()
            axes_data.append(axis_dict)

        # Save metadata as JSON
        metadata = {
            "quantity": self.quantity,
            "unit": self.unit,
            "axes": axes_data,
        }

        # Open zarr group to save metadata
        root = zarr.open_group(str(filepath), mode="a")
        root.attrs["metadata"] = json.dumps(metadata)

    @classmethod
    def load(cls, filepath: str | Path, lazy: bool = True) -> Self:
        """Load DataBlock from disk with optional lazy loading.

        Args:
            filepath: str or Path; path to the saved DataBlock
            lazy: bool; if True, data is loaded lazily using dask

        Returns:
            DataBlock; reconstructed DataBlock object
        """
        filepath = Path(filepath)
        if filepath.suffix != ".zarr":
            filepath = filepath.with_suffix(".zarr")

        # Open zarr group
        root = zarr.open_group(str(filepath), mode="r")

        # Load metadata
        metadata = json.loads(root.attrs["metadata"])

        # Load data
        data_path = str(filepath / "data")
        if lazy:
            data = da.from_zarr(data_path)
        else:
            zarr_array = zarr.open_array(data_path, mode="r")
            data = da.from_array(zarr_array[:])

        # Reconstruct axes
        axes = []
        for axis_dict in metadata["axes"]:
            axis_type = axis_dict["type"]
            if axis_type == "SignalAxis":
                axis = SignalAxis(
                    axis_points=np.array(axis_dict["points"]),
                    name=axis_dict["name"],
                    index_in_array=axis_dict["index_in_array"],
                    unit=axis_dict["unit"],
                    navigate=axis_dict["navigate"],
                )
            elif axis_type == "UnorderedSignalAxis":
                axis = UnorderedSignalAxis(
                    axis_points=np.array(axis_dict["points"]),
                    name=axis_dict["name"],
                    index_in_array=axis_dict["index_in_array"],
                    unit=axis_dict["unit"],
                    navigate=axis_dict["navigate"],
                )
            elif axis_type == "CategoricalAxis":
                axis = CategoricalAxis(
                    categories=axis_dict["categories"],
                    name=axis_dict["name"],
                    index_in_array=axis_dict["index_in_array"],
                    unit=axis_dict["unit"],
                    navigate=axis_dict["navigate"],
                )
            else:
                raise ValueError(f"Unknown axis type: {axis_type}")
            axes.append(axis)

        return cls(
            data=data,
            axes=axes,
            quantity=metadata["quantity"],
            unit=metadata["unit"],
        )

    def iter_axis(
        self, axis_name: str, min_idx: Optional[int] = None, max_idx: Optional[int] = None
    ) -> Generator:
        if self.has_axis(axis_name):
            axis_idx = self.axis_obj(axis_name).index_in_array
            _data = self.data.copy()
            if min_idx is not None and max_idx is not None:
                _data = np.transpose(_data, (axis_idx, -1))[min_idx:max_idx]
            else:
                _data = np.transpose(_data, (axis_idx, -1))
            for i in range(_data.shape[0]):
                yield _data[i]
        else:
            raise RuntimeError(f"Axis {axis_name} not found in {self.axes}")

    def quantity_data(self) -> da.Array:
        return self.data

    def has_axis(self, axis_name: str) -> bool:
        """returns True if datablock has axis with `axis_name`

        Args:
            axis_name: str;

        Returns:
            bool;
        """
        for axis in self.axes:
            if axis.name == axis_name:
                return True
        return False

    def axis_obj(self, axis_name: str) -> AxisLike:
        """Returns axis corresponding to name `axis_name`

        Args:
            axis_name: str; name of the to be returned axis

        Returns:
            AxisLike; requested axis object

        Raises:
            RuntimeError: If datablock does not have axis with name `axis_name`
        """
        if self.has_axis(axis_name):
            for axis in self.axes:
                if axis.name == axis_name:
                    return axis
        raise RuntimeError(f"{self} does not have axis {axis_name}")

    def axis_points_by_name(self, axis_name: str) -> np.ndarray:
        """Return 1D-array of points corresponding to axis with name `axis_name`

        Args:
            axis_name: str; name of axis for which points are requested

        Returns:
            np.ndarray; points of axis with name `axis_name`

        Raises:
            RuntimeError: If datablock does not have axis with name `axis_name`
        """
        if self.has_axis(axis_name):
            for axis in self.axes:
                if axis.name == axis_name:
                    return axis.points
        raise RuntimeError(f"{self} does not have axis {axis_name}")

    def __sub__(self, other):
        new_data = self.data - other
        return DataBlock(data=new_data, axes=self.axes)

    def __add__(self, other):
        new_data = self.data + other
        return DataBlock(data=new_data, axes=self.axes)

    def __mul__(self, other):
        new_data = self.data * other
        return DataBlock(data=new_data, axes=self.axes)

    def __pow__(self, other):
        new_data = self.data**other
        return DataBlock(data=new_data, axes=self.axes)

    def crop_axis(self, axis_name: str, min: float, max: float) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"Axis {axis_name} not found")

        if isinstance(self.axis_obj(axis_name), CategoricalAxis):
            raise RuntimeError(f"Method not implemented for axis CategoricalAxis {axis_name}")

        relevant_axis: SignalAxis | UnorderedSignalAxis = self.axis_obj(axis_name)

        coords = relevant_axis.points
        mask = (coords >= min) & (coords <= max)
        keep_values = da.compress(
            condition=mask,
            a=self.data,
            axis=relevant_axis.index_in_array,
        )
        new_axis = SignalAxis(
            axis_points=coords[mask],
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )

        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = new_axis
        return type(self)(keep_values, new_axes, self.quantity, self.unit)

    def rebin_reduce_axis(
        self, axis_name: str, new_binsize: Number, reducer: Callable = da.mean
    ) -> Self:
        """Rebins axis with `axis_name` such that the new spacing of the points
        of the axis are roughly `new_binsize`. The multiple datapoints in the
        new bin will be reduced to a single datapoint by calling the function
        `reducer`, by default this computes the mean.

        Args:
            axis_name: str; name of axis to reduce
            new_binsize: Number; spacing of the reduced axis
            reducer: Callable; function that reduces n points to 1 point

        Returns:
            DataBlock; datablock with reduced size along one axis
        """
        if not self.has_axis(axis_name):
            raise RuntimeError(f"Axis {axis_name} not found")
        for axis in self.axes:
            if axis.name == axis_name:
                relevant_axis: AxisLike = axis
        if not new_binsize > relevant_axis.scale:
            raise RuntimeError(
                f"Rebin to reduce, new binsize {new_binsize} is not smaller than current axis spacing {relevant_axis.scale}"
            )

        num_bins = int(np.floor((relevant_axis.max - relevant_axis.min) / new_binsize))
        bin_edges = np.linspace(relevant_axis.min, relevant_axis.max, num_bins) - new_binsize / 2
        bin_centr = np.linspace(relevant_axis.min, relevant_axis.max, num_bins)

        groups = np.digitize(relevant_axis.points, bins=bin_edges, right=True)

        new_axis = SignalAxis(
            axis_points=bin_centr,
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )

        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = new_axis

        new_view = da.stack(
            seq=[
                reducer(
                    da.take(self.data, indices=(groups == i), axis=relevant_axis.index_in_array),
                    axis=new_axis.index_in_array,
                )
                for i in np.unique(groups)
            ],
            axis=new_axis.index_in_array,
        )
        return type(self)(new_view, new_axes)

    def reduce_axis(
        self,
        axis_name: str,
        reducer: Callable[
            np.ndarray[tuple[Literal[1]], np.dtype[np.floating | np.integer]], Number
        ] = lambda s: s.max(),
        out_quantity: str = "q",
        out_unit: str = "-",
    ) -> Self:
        """Fully reduces axis with name `axis_name`. Default reducer is .max()
        works similarily to np.max(array, axis=n) but with named axis.

        Args:
            axis_name: str; name of axis to reduce
            reducer: Callable; function to reduce array to point
            out_quantity: str; name of quantity in returned datablock
            out_unit: str; name of the unit of quantity

        Returns:
            [TODO:return]
        """
        if not self.has_axis(axis_name):
            raise RuntimeError(f"Axis {axis_name} not found")

        relevant_axis: AxisLike = self.axis_obj(axis_name)

        new_axes = deepcopy(self.axes)
        new_axes.pop(relevant_axis.index_in_array)
        # TODO: Handle popping last axis
        for i in range(len(new_axes)):
            new_axes[i].index_in_array = i

        new_data = da.apply_along_axis(
            func1d=reducer, axis=relevant_axis.index_in_array, arr=self.data
        )

        return type(self)(new_data, new_axes, quantity=out_quantity, unit=out_unit)

    def get(self, indexing: dict) -> Self:
        """
        Advanced indexing method using dictionary specification.

        Args:
            indexing: Dictionary mapping axis names to either:
                     - Inequality string (numeric axes): 'x<3', 'x>2', '-3<=x<3'
                     - Equality string (categorical axes): 'category==dog', 'label!=positive'
                     - Boolean array/list: np.array([True, False, ...])

        Axes not mentioned in the dict are not indexed (all values kept).

        Examples:
            datablock.get({'x': 'x>2', 'y': 'y<1', 'e': '-3<=e<3'})
            datablock.get({'category': 'category==dog'})
            datablock.get({'x': np.array([False, True, True, ...])})
            datablock.get({'x': 'x>0', 'y': [True, False, True]})

        Returns:
            DataBlock: New DataBlock with filtered data and updated axes
        """
        if not isinstance(indexing, dict):
            raise TypeError(f"Expected dict, got {type(indexing)}")

        if len(indexing) == 0:
            return self

        # Start with current data and axes
        new_data = self.data
        new_axes = deepcopy(self.axes)

        # Process each axis in the indexing dict
        for axis_name, selector in indexing.items():
            if not self.has_axis(axis_name):
                raise ValueError(f"Axis '{axis_name}' not found in DataBlock")

            axis_obj = self.axis_obj(axis_name)
            axis_idx = axis_obj.index_in_array

            # Case 1: String expression
            if isinstance(selector, str):
                print(axis_obj, type(axis_obj))
                if isinstance(axis_obj, CategoricalAxis):
                    print('is categorical')
                    mask = parse_categorical_to_mask(axis_name, selector, axis_obj.points)
                else:
                    print('not categorical')
                    mask = parse_inequality_to_mask(axis_name, selector, axis_obj.points)

            # Case 2: Boolean array or list
            elif isinstance(selector, (list, np.ndarray, da.Array)):
                mask = np.array(selector) if isinstance(selector, list) else selector

                if len(mask) != axis_obj.size:
                    raise ValueError(
                        f"Mask length ({len(mask)}) doesn't match axis '{axis_name}' size ({axis_obj.size})"
                    )
            else:
                raise TypeError(
                    f"Selector for axis '{axis_name}' must be string or boolean array, got {type(selector)}"
                )

            # Apply mask to data
            new_data = da.compress(condition=mask, a=new_data, axis=axis_idx)

            # Update corresponding axis
            if isinstance(axis_obj, CategoricalAxis):
                new_axis = CategoricalAxis(
                    categories=axis_obj.points[mask].tolist(),
                    name=axis_obj.name,
                    index_in_array=axis_obj.index_in_array,
                    unit=axis_obj.unit,
                    navigate=axis_obj.is_nav,
                )
            else:
                new_axis = SignalAxis(
                    axis_points=axis_obj.points[mask],
                    name=axis_obj.name,
                    index_in_array=axis_obj.index_in_array,
                    unit=axis_obj.unit,
                    navigate=axis_obj.is_nav,
                )
            new_axes[axis_idx] = new_axis

        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def value_get(self, indexing: dict) -> tuple[da.Array, dict[str, np.ndarray]]:
        """
        Advanced indexing method that returns raw values instead of a DataBlock.

        Args:
            indexing: Dictionary mapping axis names to either:
                     - Inequality string (numeric axes): 'x<3', 'x>2', '-3<=x<3'
                     - Equality string (categorical axes): 'category==dog', 'label!=positive'
                     - Boolean array/list: np.array([True, False, ...])

        Axes not mentioned in the dict are not indexed (all values kept).

        Examples:
            quantity_data, axes_dict = datablock.value_get({'x': 'x>2', 'y': 'y<1'})
            # quantity_data is a dask array with the filtered data
            # axes_dict is {'x': np.array([...]), 'y': np.array([...]), ...}

        Returns:
            tuple: (quantity_data, axes_dict) where:
                - quantity_data: dask.array.Array with filtered quantity values
                - axes_dict: dict mapping axis names to their numpy arrays
        """
        if not isinstance(indexing, dict):
            raise TypeError(f"Expected dict, got {type(indexing)}")

        if len(indexing) == 0:
            # No filtering, return all data
            axes_dict = {axis.name: axis.points for axis in self.axes}
            return self.data, axes_dict

        # Start with current data and axes
        new_data = self.data
        new_axes = deepcopy(self.axes)

        # Process each axis in the indexing dict
        for axis_name, selector in indexing.items():
            if not self.has_axis(axis_name):
                raise ValueError(f"Axis '{axis_name}' not found in DataBlock")

            axis_obj = self.axis_obj(axis_name)
            axis_idx = axis_obj.index_in_array

            # Case 1: String expression
            if isinstance(selector, str):
                if isinstance(axis_obj, CategoricalAxis):
                    mask = parse_categorical_to_mask(axis_name, selector, axis_obj.points)
                else:
                    mask = parse_inequality_to_mask(axis_name, selector, axis_obj.points)

            # Case 2: Boolean array or list
            elif isinstance(selector, (list, np.ndarray, da.Array)):
                mask = np.array(selector) if isinstance(selector, list) else selector

                if len(mask) != axis_obj.size:
                    raise ValueError(
                        f"Mask length ({len(mask)}) doesn't match axis '{axis_name}' size ({axis_obj.size})"
                    )
            else:
                raise TypeError(
                    f"Selector for axis '{axis_name}' must be string or boolean array, got {type(selector)}"
                )

            # Apply mask to data
            new_data = da.compress(condition=mask, a=new_data, axis=axis_idx)

            # Update corresponding axis points
            new_axes[axis_idx] = type(axis_obj)(
                **(
                    {
                        "categories": axis_obj.points[mask].tolist(),
                        "name": axis_obj.name,
                        "index_in_array": axis_obj.index_in_array,
                        "unit": axis_obj.unit,
                        "navigate": axis_obj.is_nav,
                    }
                    if isinstance(axis_obj, CategoricalAxis)
                    else {
                        "axis_points": axis_obj.points[mask],
                        "name": axis_obj.name,
                        "index_in_array": axis_obj.index_in_array,
                        "unit": axis_obj.unit,
                        "navigate": axis_obj.is_nav,
                    }
                )
            )

        # Build axes dictionary
        axes_dict = {axis.name: axis.points for axis in new_axes}

        return new_data, axes_dict

    def cluster(self, num_clusters: int = 3) -> Self:
        """
        clusters the quantity data and returns a datablock with same axes and cluster label as data
        """
        try:
            import dask_ml.cluster
        except ImportError:
            raise ImportError(
                "dask-ml is required for clustering functionality\nPlease install via `pip install dask-ml`"
            )

        clusterer = dask_ml.cluster.KMeans(n_clusters=num_clusters)
        data_reshaped = self.data.flatten().reshape((-1, 1))
        cluster_labels = clusterer.fit(data_reshaped).predict(data_reshaped)
        cluster_data = cluster_labels.reshape(self.data.shape)
        new_axes = deepcopy(self.axes)

        return type(self)(cluster_data, new_axes, quantity="cluster", unit="-")

    def split_on_axis(self, axis_name: str) -> list[Self]:
        """
        Method will split the axis with axis_name and return a list of DataBlock.
        For example:
            3D DataBlock with axis, size equal to (x, 10), (y, 20), (E, 400)
            with split_on_axis('x') will yield a list with 10 DataBlocks all
            with size (1, 20, 400)
        """
        split_blocks: list[Self] = list()
        split_axis: AxisLike = self.axis_obj(axis_name)
        if isinstance(split_axis, UnorderedSignalAxis):
            print(
                RuntimeWarning(
                    f"WARNING: Splitting on UnorderedSignalAxis will potentially return very inefficient list[{type(self)}]"
                )
            )
        split_coords: np.ndarray | Axis1D = split_axis.points
        for val in split_coords:
            new_axis = CategoricalAxis(
                categories=[f"{val:.3f}"],
                name=split_axis.name,
                index_in_array=split_axis.index_in_array,
                unit=split_axis.unit,
                navigate=False,
            )
            new_axes: list[AxisLike] = deepcopy(self.axes)
            new_axes[new_axis.index_in_array] = new_axis
            new_data = da.take(
                self.data, indices=(split_coords == val), axis=new_axis.index_in_array
            )

            split_blocks.append(
                type(self)(data=new_data, axes=new_axes, quantity=self.quantity, unit=self.unit)
            )
        return split_blocks

    def split_with_datablock(
        self, other: Self, drop_axes: list[str] | str | None = None
    ) -> list[Self]:
        """
        Method will split the DataBlock using the values of other as a mask.
        Such that calling split_with_datablock with other equal to a datablock
        with 10 unique values will split this datablock into 10.

        The axes in drop_axes are optionally removed from the axes of the
        final list of datablocks to reduce sparsity.
        If `drop_axes` is None then the output datablocks will retain their shape
        but values not in the cluster will be masked with `np.nan`.
        """
        if drop_axes is None:
            _drop_any: bool = False
        elif isinstance(drop_axes, str):
            _drop_any: bool = True
            drop_axes = [drop_axes]
        elif isinstance(drop_axes, list) and len(drop_axes) == 0:
            _drop_any = False
        elif isinstance(drop_axes, list) and len(drop_axes) != 0:
            _drop_any = True
        else:
            raise ValueError(f"drop_axes should be list[str], str or None.\nGot {type(drop_axes)}")

        for axis in other.axes:
            if not self.has_axis(axis.name):
                raise RuntimeError(
                    f"Split on datablock has axes unkown to the to be splitted datablock\n\t{axis.name} not found for {self}"
                )

        missing = []
        for axis in self.axes:
            if not other.has_axis(axis.name):
                missing.append(axis.index_in_array)

        if _drop_any:
            new_axes: list[AxisLike] = []
            for axis in self.axes:
                if axis.name not in drop_axes:
                    new_axes.append(deepcopy(axis))
        elif not _drop_any:
            new_axes = deepcopy(self.axes)

        for i, axis in enumerate(new_axes):
            axis.index_in_array = i

        new_shape = tuple([axis.size for axis in new_axes])

        split_blocks: list[Self] = list()
        unique_values = da.unique(other.quantity_data()).compute()
        unique_values = np.sort(unique_values)
        for uval in unique_values:
            mask = other.quantity_data() == uval
            mask = da.expand_dims(mask, axis=tuple(missing[::-1]))
            mask = da.broadcast_to(mask, self.data.shape)

            if _drop_any:
                matching = self.data[mask]
                matching.compute_chunk_sizes()
                new_data = matching.reshape(new_shape + (-1,)).rechunk({0: -1})
                new_axis = SignalAxis(
                    axis_points=np.arange(new_data.shape[-1]),
                    name="repetition",
                    index_in_array=len(new_axes),
                    unit="#",
                    navigate=False,
                )
                split_blocks.append(
                    type(self)(
                        new_data, new_axes + [new_axis], quantity=self.quantity, unit=self.unit
                    )
                )

            elif not _drop_any:
                new_data = da.where(mask, self.data, da.ones(self.data.shape) * np.nan)
                split_blocks.append(
                    type(self)(new_data, new_axes, quantity=self.quantity, unit=self.unit)
                )

            else:
                raise RuntimeError(
                    "Code should be unreachable, check `drop_axes` input matches signature"
                )
        return split_blocks

    def to_ensemble(self):
        return Ensemble.from_datablock(datablock=self)


class Ensemble:
    """
    Container for data that does not nicely conform to a coordinate grid
    """

    def __init__(
        self,
        data: dd.DataFrame,
        axes: list[UnorderedSignalAxis | SignalAxis],
        quantity: str = "q",
        unit: str = "-",
    ):
        self.data: dd.DataFrame = data
        self.axes: list[UnorderedSignalAxis | SignalAxis] = axes
        self.dims: int = len(axes)
        self.quantity: str = quantity
        self.unit: str = unit
        if isinstance(self.data, dd.DataFrame):
            self._computed: bool = False
        else:
            self._computed = True

    def compute(self) -> pd.DataFrame:
        if not self._computed:
            self.data = self.data.compute()
            self._computed = True
        return self.data

    def save(self, filepath: str | Path) -> None:
        """Save Ensemble to disk using Zarr/Parquet format for lazy loading.

        Args:
            filepath: str or Path; path where the Ensemble will be saved
        """
        filepath = Path(filepath)
        if filepath.suffix != ".zarr":
            filepath = filepath.with_suffix(".zarr")

        # Create zarr group
        root = zarr.open_group(str(filepath), mode="w")

        # Save dataframe as parquet inside zarr directory
        parquet_path = filepath / "dataframe.parquet"
        self.data.to_parquet(str(parquet_path), engine="pyarrow")

        # Save axes information
        axes_data = []
        for axis in self.axes:
            axis_dict = {
                "type": type(axis).__name__,
                "name": axis.name,
                "index_in_array": axis.index_in_array,
                "unit": axis.unit,
                "navigate": axis.is_nav,
            }
            if isinstance(axis, (SignalAxis, UnorderedSignalAxis)):
                axis_dict["points"] = (
                    axis.points.tolist() if hasattr(axis.points, "tolist") else list(axis.points)
                )
            axes_data.append(axis_dict)

        # Save metadata as JSON
        metadata = {
            "quantity": self.quantity,
            "unit": self.unit,
            "axes": axes_data,
        }
        root.attrs["metadata"] = json.dumps(metadata)

    @classmethod
    def load(cls, filepath: str | Path, lazy: bool = True) -> Self:
        """Load Ensemble from disk with optional lazy loading.

        Args:
            filepath: str or Path; path to the saved Ensemble
            lazy: bool; if True, data is loaded lazily using dask

        Returns:
            Ensemble; reconstructed Ensemble object
        """
        filepath = Path(filepath)
        if filepath.suffix != ".zarr":
            filepath = filepath.with_suffix(".zarr")

        # Open zarr group
        root = zarr.open_group(str(filepath), mode="r")

        # Load metadata
        metadata = json.loads(root.attrs["metadata"])

        # Load dataframe
        parquet_path = filepath / "dataframe.parquet"
        if lazy:
            data = dd.read_parquet(str(parquet_path), engine="pyarrow")
        else:
            data = dd.from_pandas(
                pd.read_parquet(str(parquet_path), engine="pyarrow"), npartitions=1
            )

        # Reconstruct axes
        axes = []
        for axis_dict in metadata["axes"]:
            axis_type = axis_dict["type"]
            if axis_type == "SignalAxis":
                axis = SignalAxis(
                    axis_points=np.array(axis_dict["points"]),
                    name=axis_dict["name"],
                    index_in_array=axis_dict["index_in_array"],
                    unit=axis_dict["unit"],
                    navigate=axis_dict["navigate"],
                )
            elif axis_type == "UnorderedSignalAxis":
                axis = UnorderedSignalAxis(
                    axis_points=np.array(axis_dict["points"]),
                    name=axis_dict["name"],
                    index_in_array=axis_dict["index_in_array"],
                    unit=axis_dict["unit"],
                    navigate=axis_dict["navigate"],
                )
            else:
                raise ValueError(f"Unknown axis type: {axis_type}")
            axes.append(axis)

        return cls(
            data=data,
            axes=axes,
            quantity=metadata["quantity"],
            unit=metadata["unit"],
        )

    @classmethod
    def from_datablock(cls, datablock: DataBlock) -> Self:  # type: ignore
        data = datablock.data.flatten()
        axes = datablock.axes
        quantity = datablock.quantity
        unit = datablock.unit
        mgrid = da.meshgrid(*[axis.points for axis in axes], indexing="ij")

        new_axes = []
        for i, axis in enumerate(datablock.axes):
            new_axis = SignalAxis(
                axis_points=axis.points,
                name=axis.name,
                index_in_array=axis.index_in_array,
                unit=axis.unit,
                navigate=axis.is_nav,
            )
            new_axis.scale = axis.scale
            new_axes.append(new_axis)
        axes = new_axes
        coord_block = (
            da.stack([mgrid[i].flatten() for i in range(len(axes))] + [data])
            .rechunk({0: -1, 1: "auto"})
            .T
        )
        data = dd.io.from_dask_array(coord_block, columns=[axis.name for axis in axes] + [quantity])
        return Ensemble(data, axes, quantity, unit)  # type: ignore

    def __sub__(self, other) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] = new_data[self.quantity] - other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __add__(self, other) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] = new_data[self.quantity] + other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __mul__(self, other) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] = new_data[self.quantity] * other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __pow__(self, other) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] = new_data[self.quantity] ** other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def has_axis(self, axis_name: str) -> bool:
        for axis in self.axes:
            if axis.name == axis_name:
                return True
        return False

    def axis_obj(self, axis_name: str) -> SignalAxis | UnorderedSignalAxis:
        if not self.has_axis(axis_name):
            raise ValueError(f"{self} has no axis: {axis_name}")
        else:
            for axis in self.axes:
                if axis.name == axis_name:
                    return axis
        raise RuntimeError("Should not reach here")

    def axis(self, axis_name: str) -> Axis1D:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        for axis in self.axes:
            if axis.name == axis_name:
                return axis.points

    def reduce_axis(
        self,
        axis_name: str,
        reducer: dd.groupby.Aggregation,
        out_quantity: str = "q",
        out_unit: str = "-",
    ) -> Self:
        """Fully reduces axis with name `axis_name`. Default reducer is .max()
        works similarily to np.max(array, axis=n) but with named axis.

        Args:
            axis_name: str; name of axis to reduce
            reducer: Callable; function to reduce array to point
            out_quantity: str; name of quantity in returned datablock
            out_unit: str; name of the unit of quantity

        Returns:
            [TODO:return]
        """
        if not self.has_axis(axis_name):
            raise RuntimeError(f"Axis {axis_name} not found")

        for axis in self.axes:
            if axis.name == axis_name:
                relevant_axis: AxisLike = axis

        new_axes = deepcopy(self.axes)
        new_axes.pop(relevant_axis.index_in_array)
        # TODO: Handle popping last axis
        for i in range(len(new_axes)):
            new_axes[i].index_in_array = i

        sort_axes = [axis.name for axis in self.axes]
        sort_axes.remove(relevant_axis.name)

        new_view = (
            self.data.groupby(sort_axes, observed=True)[self.quantity]
            .agg(reducer)
            .reset_index()
        )

        return type(self)(new_view, new_axes, quantity=out_quantity, unit=out_unit)

    def rebin_reduce_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Sequence[Number] | None = None,
        reducer: Callable = da.mean,
    ) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")
        for axis in self.axes:
            if axis.name == axis_name:
                relevant_axis: AxisLike = axis

        if bin_edges is None and new_binsize is not None:
            num_bins = int(np.floor((relevant_axis.max - relevant_axis.min + 0.02) / new_binsize))
            bin_edges_: np.ndarray[tuple[Literal[1],], np.dtype[np.float64 | np.int16]] = (
                np.linspace(
                    relevant_axis.min - 0.01 - new_binsize,
                    relevant_axis.max + 0.01 + new_binsize,
                    num_bins,
                )
            )
            bin_centers = np.linspace(
                relevant_axis.min - 0.01 + new_binsize / 2,
                relevant_axis.max + 0.01 - new_binsize / 2,
                num_bins - 1,
            )
        elif bin_edges is not None:
            bin_edges_ = np.array(bin_edges)
            bin_centers = bin_edges[1:] - (bin_edges_[1:] - bin_edges_[:-1])/2

        rebin_axis = relevant_axis.name + "_bin"
        sort_axes = [axis.name for axis in self.axes] + [rebin_axis]
        sort_axes.remove(relevant_axis.name)

        new_view = self.data.assign(
            **{
                rebin_axis: self.data[relevant_axis.name].map_partitions(
                    pd.cut, bins=bin_edges_, include_lowest=True, labels=False
                )
            }
        )
        new_view = new_view.dropna()
        new_view = (
            new_view.groupby(sort_axes, observed=True)[self.quantity]
            .agg(list, meta=(self.quantity, "float64"))
            .reset_index()
        )

        new_view[self.quantity] = new_view[self.quantity].apply(
            reducer, meta=(relevant_axis.name, "float64")
        )

        def _mapper_func(i):
            if np.isnan(i):
                return 1_000_000
            return bin_centers[int(i)]

        new_view[relevant_axis.name] = new_view[rebin_axis].apply(
            _mapper_func, meta=(relevant_axis.name, "float64")
        )
        new_view = new_view.drop(columns=rebin_axis)

        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = SignalAxis(
            axis_points=bin_centers,
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )

        return type(self)(new_view, new_axes, self.quantity, self.unit)

    def rebin_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Sequence[Number] | None = None,
    ) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        assert new_binsize is not None or bin_edges is not None, (
            "Either call with new_binsize= or bin_edges="
        )

        for axis in self.axes:
            if axis.name == axis_name:
                relevant_axis: AxisLike = axis

        if bin_edges is None and new_binsize is not None:
            num_bins = int(np.floor((relevant_axis.max - relevant_axis.min + 0.02) / new_binsize))
            bin_edges_: np.ndarray[tuple[Literal[1],], np.dtype[np.float64 | np.int16]] = (
                np.linspace(
                    relevant_axis.min - 0.01 - new_binsize,
                    relevant_axis.max + 0.01 + new_binsize,
                    num_bins,
                )
            )
            bin_centers = np.linspace(
                relevant_axis.min - 0.01 + new_binsize / 2,
                relevant_axis.max + 0.01 - new_binsize / 2,
                num_bins - 1,
            )
        elif bin_edges is not None:
            bin_edges_ = np.array(bin_edges)
            bin_centers = bin_edges[1:] - (bin_edges_[1:] - bin_edges_[:-1])/2

        rebin_axis = relevant_axis.name + "_bin"
        sort_axes = [axis.name for axis in self.axes] + [rebin_axis]
        sort_axes.remove(relevant_axis.name)

        new_view = self.data.assign(
            **{
                rebin_axis: self.data[relevant_axis.name].map_partitions(
                    pd.cut, bins=bin_edges_, include_lowest=True, labels=False
                )
            }
        )

        new_view = new_view.dropna()
        # new_view = (
        #     new_view.groupby(sort_axes, observed=True)[self.quantity]
        #     .agg(list, meta=(self.quantity, "object"))
        #     .reset_index()
        # )
        def _mapper_func(i):
            if np.isnan(i):
                return 1_000_000
            return bin_centers[int(i)]

        new_view[relevant_axis.name] = new_view[rebin_axis].apply(
            _mapper_func, meta=(relevant_axis.name, "float64")
        )
        new_view = new_view.drop(columns=rebin_axis)

        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = SignalAxis(
            axis_points=bin_centers,
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )

        return type(self)(new_view, new_axes, self.quantity, self.unit)

    def rebin_to_new_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Sequence[Number] | None = None,
    ) -> Self:
        """Rebin axis to discrete values and add repetition counter axis.

        This method rebins the specified axis to discrete bin centers (like rebin_axis)
        but also adds a new axis that counts repetitions within each bin. Points that
        fall into the same bin and have identical coordinates on all other axes will
        be assigned increasing repetition numbers (0, 1, 2, ...). The quantity values
        remain unchanged.

        Args:
            axis_name: Name of the axis to rebin
            new_binsize: Size of the new bins (either this or bin_edges must be provided)
            bin_edges: Explicit bin edges (either this or new_binsize must be provided)

        Returns:
            New Ensemble with rebinned axis and added repetition axis
        """
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        assert new_binsize is not None or bin_edges is not None, (
            "Either call with new_binsize= or bin_edges="
        )
        for axis in self.axes:
            if axis.name == axis_name:
                relevant_axis: AxisLike = axis

        if bin_edges is None and new_binsize is not None:
            num_bins = int(np.floor((relevant_axis.max - relevant_axis.min + 0.02) / new_binsize))
            bin_edges_: np.ndarray[tuple[Literal[1],], np.dtype[np.float64 | np.int16]] = (
                np.linspace(
                    relevant_axis.min - 0.01 - new_binsize,
                    relevant_axis.max + 0.01 + new_binsize,
                    num_bins,
                )
            )
            bin_centers = np.linspace(
                relevant_axis.min - 0.01 + new_binsize / 2,
                relevant_axis.max + 0.01 - new_binsize / 2,
                num_bins - 1,
            )
        elif bin_edges is not None:
            bin_edges_ = np.array(bin_edges)
            bin_centers = bin_edges_[1:] - bin_edges_[:-1]

        rebin_axis = relevant_axis.name + "_bin"

        # Assign bin indices
        new_view = self.data.assign(
            **{
                rebin_axis: self.data[relevant_axis.name].map_partitions(
                    pd.cut, bins=bin_edges_, include_lowest=True, labels=False
                )
            }
        )

        # Replace axis values with bin centers
        def _mapper_func(i):
            if np.isnan(i):
                return 1_000_000
            return bin_centers[int(i)]

        new_view[relevant_axis.name] = new_view[rebin_axis].apply(
            _mapper_func, meta=(relevant_axis.name, "float64")
        )
        new_view = new_view.drop(columns=rebin_axis)

        # Create groupby keys for all axes to identify unique coordinate combinations
        group_axes = [ax.name for ax in self.axes]

        # Add repetition counter within each unique coordinate combination
        new_view["repetition"] = new_view.groupby(group_axes, observed=True).cumcount()

        # Drop the temporary bin column
        new_view = new_view.drop(columns=rebin_axis)

        # Create new axes list with the repetition axis
        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = SignalAxis(
            axis_points=bin_centers,
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )

        # Add the repetition axis as an SignalAxis
        rep_axis = SignalAxis(
            axis_points=da.unique(new_view["repetition"].to_dask_array()).compute(),
            name="repetition",
            index_in_array=len(new_axes),
            unit="#",
            navigate=False,
        )
        new_axes.append(rep_axis)

        return type(self)(new_view, new_axes, self.quantity, self.unit)

    def to_datablock(self, rebin_scheme: dict[str, Number] = {}) -> DataBlock:
        new_axes: list[SignalAxis] = []
        idx_axes: list[da.Array | np.ndarray] = []
        for axis in self.axes:
            if isinstance(axis, SignalAxis):
                naxis_points = self.data[axis.name].astype(float)
                naxis_min = axis.min
                naxis_psp = axis.scale

                naxis_pidx = (
                    ((naxis_points - naxis_min) / naxis_psp).astype(int).to_dask_array(lengths=True)
                )
                idx_axes.append(naxis_pidx)
                new_axes.append(axis)
            elif isinstance(axis, UnorderedSignalAxis) and axis.name in rebin_scheme.keys():
                raise NotImplementedError()

            else:
                raise RuntimeError(f"Can not create dense block with floating axes: {axis.name}")

        values = self.data[self.quantity].to_dask_array(lengths=True)
        coords = da.stack(idx_axes, axis=0).rechunk((values.chunksize[0], -1))
        nshape = tuple([ax.size for ax in new_axes])
        print(coords, values, nshape)

        nanmask = ~(da.isnan(coords) & da.isnan(values)[None, :])
        nanmask_flat = da.all(nanmask, axis=0)
        coords = da.take(coords, nanmask_flat, axis=1).compute_chunk_sizes()
        values = values[nanmask_flat].compute_chunk_sizes()
        print(nanmask.shape, coords.shape, values.shape)

        data_coo = da.map_blocks(
            lambda c, v: sparse.COO(c, v, shape=nshape, fill_value=np.nan),
            coords,
            values,
            dtype=values.dtype,
            # drop_axis=range(len(coords.shape)),
            # new_axis=range(len(nshape)),
        )
        print(f'data_coo: [shape: {data_coo.shape}, # axes: {len(new_axes)}]')
        data_db = data_coo.map_blocks(
            lambda b: b.todense(),
            dtype=values.dtype,
        )
        print(f'data_db: [shape: {data_db.shape}, # axes: {len(new_axes)}]')
        print('!!'+'+'*10 + 'Forcing compute' + '+'*10 + '!!')
        data_db = data_db.compute() # FIXME: Somehow this should not be necesarry

        return DataBlock(data_db, new_axes, self.quantity, self.unit)

    def rename_quantity(self, new_name: str) -> Self:
        old_name = self.quantity
        new_data = self.data.rename(columns={old_name: new_name})
        return type(self)(new_data, self.axes, new_name, self.unit)

    def c_op(self, other: Self, operation: TwoSeriesOperation) -> Self:
        """
        operation on any column of the `self.data` dataframe
        where self.data[quantity] operation other.data[quantity] is
        performed.
        left and right are joined based on the axis values of self
        and other
        """
        assert other.quantity in self.data.columns, (
            f"Can not perform operation, self.data has not column {other.quantity}"
        )

        self_data: pd.DataFrame = self.data
        other_data: pd.DataFrame = other.data

        left_axes = [ax.name for ax in self.axes]
        right_axes = [ax.name for ax in other.axes]
        common_axes = list(set(left_axes) & set(right_axes))

        new_data = self_data.merge(other_data, how="left", on=common_axes, suffixes=("", "_other"))
        new_data[other.quantity] = operation(
            new_data[other.quantity], new_data[other.quantity + "_other"]
        )
        new_data = new_data.drop(columns=other.quantity + "_other")

        if other.quantity in left_axes:
            old_axis: SignalAxis | UnorderedSignalAxis = self.axes[left_axes.index(other.quantity)]
            new_axis = UnorderedSignalAxis(
                axis_points=new_data[other.quantity],
                name=old_axis.name,
                index_in_array=old_axis.index_in_array,
                unit=old_axis.unit,
                navigate=old_axis.is_nav,
            )
            new_axes = [new_axis if ax.name == other.quantity else ax for ax in self.axes]
        else:
            new_axes = deepcopy(self.axes)

        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def combine(self, other: Self) -> Self:
        self_data: pd.DataFrame = self.data
        other_data: pd.DataFrame = other.data

        left_axes = [ax.name for ax in self.axes]
        right_axes = [ax.name for ax in other.axes]
        common_axes = list(set(left_axes) & set(right_axes))

        new_data = self_data.merge(other_data, how="left", on=common_axes, suffixes=("", "_other"))
        old_axes = self.axes
        new_axis = UnorderedSignalAxis(
            axis_points=other.data[other.quantity],
            name=other.quantity,
            index_in_array=len(self.axes),
            unit=other.unit,
            navigate=False,
        )
        return type(self)(new_data, old_axes + [new_axis], self.quantity, self.unit)

    def get(self, indexing: dict) -> Self:
        """
        Advanced indexing method using dictionary specification.

        Args:
            indexing: Dictionary mapping axis names to either:
                     - Inequality string (numeric axes): 'x<3', 'x>2', '-3<=x<3'
                     - Equality string (categorical axes): 'category==dog', 'label!=positive'
                     - Boolean array/list: np.array([True, False, ...])

        Axes not mentioned in the dict are not indexed (all values kept).

        Examples:
            ensemble.get({'x': 'x>2', 'y': 'y<1', 'e': '-3<=e<3'})
            ensemble.get({'category': 'category==dog'})
            ensemble.get({'x': np.array([True, False, True, ...])})
            ensemble.get({'x': 'x>0', 'y': [True, False, True]})

        Returns:
            Ensemble: New Ensemble with filtered data
        """
        if not isinstance(indexing, dict):
            raise TypeError(f"Expected dict, got {type(indexing)}")

        if len(indexing) == 0:
            return self

        # Start with current data
        new_data = self.data

        # Process each axis in the indexing dict
        for axis_name, selector in indexing.items():
            if not self.has_axis(axis_name):
                raise ValueError(f"Axis '{axis_name}' not found in Ensemble")

            axis_obj = self.axis_obj(axis_name)

            # Case 1: String expression
            if isinstance(selector, str):
                if isinstance(axis_obj, CategoricalAxis):
                    query_str = parse_categorical_to_query(axis_name, selector)
                else:
                    query_str = parse_inequality_to_query(axis_name, selector)
                new_data = new_data.query(query_str)

            # Case 2: Boolean array or list
            elif isinstance(selector, (list, np.ndarray, da.Array)):
                mask = np.array(selector) if isinstance(selector, list) else selector

                # For ensemble, we need to create a mask based on axis values
                # Get unique values in the axis column
                axis_values = axis_obj.points

                if len(mask) != len(axis_values):
                    raise ValueError(
                        f"Mask length ({len(mask)}) doesn't match axis '{axis_name}' unique values ({len(axis_values)})"
                    )

                # Create a mapping from axis value to boolean
                keep_values = axis_values[mask]
                new_data = new_data[new_data[axis_name].isin(keep_values)]
            else:
                raise TypeError(
                    f"Selector for axis '{axis_name}' must be string or boolean array, got {type(selector)}"
                )

        # Update axes based on filtered data
        new_axes = []
        for axis in self.axes:
            if axis.name in indexing:
                # Get unique values from filtered data for this axis
                unique_vals = new_data[axis.name].unique()
                # if isinstance(unique_vals, da.Array):
                #     unique_vals = unique_vals.compute()

                if isinstance(axis, CategoricalAxis):
                    new_axis = CategoricalAxis(
                        categories=unique_vals.tolist(),
                        name=axis.name,
                        index_in_array=axis.index_in_array,
                        unit=axis.unit,
                        navigate=axis.is_nav,
                    )
                elif isinstance(axis, SignalAxis):
                    unique_vals = unique_vals  # Ensure sorted order
                    new_axis = SignalAxis(
                        axis_points=unique_vals,
                        name=axis.name,
                        index_in_array=axis.index_in_array,
                        unit=axis.unit,
                        navigate=axis.is_nav,
                    )
                else:  # UnorderedSignalAxis
                    unique_vals = unique_vals  # Ensure sorted order
                    new_axis = UnorderedSignalAxis(
                        axis_points=unique_vals,
                        name=axis.name,
                        index_in_array=axis.index_in_array,
                        unit=axis.unit,
                        navigate=axis.is_nav,
                    )
                new_axes.append(new_axis)
            else:
                # Axis not filtered, keep as is
                new_axes.append(axis)

        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def value_get(self, indexing: dict) -> tuple[np.ndarray | da.Array, dict[str, np.ndarray]]:
        """
        Advanced indexing method that returns raw values instead of an Ensemble.

        Args:
            indexing: Dictionary mapping axis names to either:
                     - Inequality string (numeric axes): 'x<3', 'x>2', '-3<=x<3'
                     - Equality string (categorical axes): 'category==dog', 'label!=positive'
                     - Boolean array/list: np.array([True, False, ...])

        Axes not mentioned in the dict are not indexed (all values kept).

        Examples:
            quantity_values, axes_dict = ensemble.value_get({'x': 'x>2', 'y': 'y<1'})
            # quantity_values is a numpy/dask array with the filtered quantity values
            # axes_dict is {'x': np.array([...]), 'y': np.array([...]), ...}

        Returns:
            tuple: (quantity_values, axes_dict) where:
                - quantity_values: numpy/dask array with filtered quantity values
                - axes_dict: dict mapping axis names to their numpy arrays from DataFrame columns
        """
        if not isinstance(indexing, dict):
            raise TypeError(f"Expected dict, got {type(indexing)}")

        # Start with current data
        new_data = self.data

        if len(indexing) == 0:
            # No filtering, extract values directly from DataFrame
            axes_dict = {axis.name: new_data[axis.name] for axis in self.axes}
            quantity_values = new_data[self.quantity].values
            return quantity_values, axes_dict

        # Process each axis in the indexing dict
        for axis_name, selector in indexing.items():
            if not self.has_axis(axis_name):
                raise ValueError(f"Axis '{axis_name}' not found in Ensemble")

            axis_obj = self.axis_obj(axis_name)

            # Case 1: String expression
            if isinstance(selector, str):
                if isinstance(axis_obj, CategoricalAxis):
                    query_str = parse_categorical_to_query(axis_name, selector)
                else:
                    query_str = parse_inequality_to_query(axis_name, selector)
                new_data = new_data.query(query_str)

            # Case 2: Boolean array or list
            elif isinstance(selector, (list, np.ndarray, da.Array)):
                mask = np.array(selector) if isinstance(selector, list) else selector

                # Get unique values directly from the DataFrame column
                axis_values = new_data[axis_name]

                # Create a mapping from axis value to boolean
                keep_values = axis_values[mask]
                new_data = new_data[new_data[axis_name].isin(keep_values)]
            else:
                raise TypeError(
                    f"Selector for axis '{axis_name}' must be string or boolean array, got {type(selector)}"
                )

        # Build axes dictionary from filtered DataFrame columns
        axes_dict = {}
        for axis in self.axes:
            axes_dict[axis.name] = new_data[axis.name].values

        # Extract quantity values from DataFrame
        quantity_values = new_data[self.quantity].values

        return quantity_values, axes_dict

    def split_on_axis(self, axis_name: str | None = None) -> list[Self]:
        """
        Method will split the axis with axis_name and return a list of Ensemble.
        For example:
            Ensemble with axes (x, 10 unique values), (y, 20), (E, 400)
            with split_on_axis('x') will yield a list with 10 Ensembles,
            each containing only rows where x equals one specific value.
        """
        if axis_name is None:
            raise ValueError("axis_name must be specified")

        if not self.has_axis(axis_name):
            raise ValueError(f"Axis '{axis_name}' not found in Ensemble")

        split_ensembles: list[Self] = list()
        split_axis: SignalAxis | UnorderedSignalAxis = self.axis_obj(axis_name)
        split_values: np.ndarray = split_axis.points

        for val in split_values:
            # Filter data to only rows with this specific axis value
            filtered_data = self.data[self.data[axis_name] == val]

            # Create new categorical axis with single value
            new_axis = CategoricalAxis(
                categories=[f"{val:.3f}"],
                name=split_axis.name,
                index_in_array=split_axis.index_in_array,
                unit=split_axis.unit,
                navigate=False,
            )

            new_axes: list[AxisLike] = deepcopy(self.axes)
            new_axes[new_axis.index_in_array] = new_axis

            split_ensembles.append(
                type(self)(
                    data=filtered_data, axes=new_axes, quantity=self.quantity, unit=self.unit
                )
            )

        return split_ensembles

    def split_with_ensemble(self, other: Self, drop_axes: list[str] | str) -> list[Self]:
        """
        Method will split the Ensemble using the values of other Ensemble as a mask.
        Such that calling split_with_ensemble with other equal to an ensemble
        with 10 unique values in its quantity will split this ensemble into 10.

        The two dataframes are joined on common axes, and then split based on
        unique values in the other ensemble's quantity column.

        The axes in drop_axes are removed from the final list of ensembles
        to reduce sparsity.
        """
        if isinstance(drop_axes, str):
            drop_axes = [drop_axes]

        # Check that all axes in other exist in self
        for axis in other.axes:
            if axis.name not in drop_axes and not self.has_axis(axis.name):
                raise AssertionError(
                    f"Split on ensemble has axes unknown to the to be split ensemble\n\t{axis.name} not found"
                )

        # Find common axes for joining
        left_axes = [ax.name for ax in self.axes]
        right_axes = [ax.name for ax in other.axes]
        common_axes = list(set(left_axes) & set(right_axes))

        if not common_axes:
            raise ValueError("No common axes found between self and other ensemble")

        # Join the two dataframes on common axes
        self_data: dd.DataFrame = self.data
        other_data: dd.DataFrame = other.data

        # Merge to align rows based on common axes
        merged_data = self_data.merge(
            other_data[[*common_axes, other.quantity]],
            how="left",
            on=common_axes,
            suffixes=("", "_mask"),
        )

        split_ensembles: list[Self] = list()

        # Get unique values from other's quantity column
        unique_values = merged_data[other.quantity].unique()
        if isinstance(unique_values, da.Array):
            unique_values = unique_values.compute()

        for uval in unique_values:
            # Prepare new axes (excluding dropped axes)
            new_axes: list[AxisLike] = []
            for axis in self.axes:
                if axis.name not in drop_axes:
                    new_axes.append(axis)

            # Filter merged data for this unique value
            filtered_data = merged_data[merged_data[other.quantity] == uval]

            for i, axis in enumerate(new_axes):
                axis.index_in_array = i
                axis.points = filtered_data[axis.name]

            # Drop the mask column and any dropped axes
            columns_to_keep = [ax.name for ax in new_axes] + [self.quantity]
            filtered_data = filtered_data[columns_to_keep]

            split_ensembles.append(
                type(self)(filtered_data, new_axes, quantity=self.quantity, unit=self.unit)
            )

        return split_ensembles
