from collections.abc import Callable, Generator, Iterable, Sequence
from copy import deepcopy
import json
import logging
import operator
from pathlib import Path
from typing import (
    Literal,
    Self,
    TypeAlias,
)

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sparse
import zarr

try:
    import dask_ml.cluster

    DASKML: bool = True
except ImportError:
    DASKML: bool = False

from .AxesStructures import CategoricalAxis, SignalAxis, UnorderedSignalAxis
from .IndexingUtils import (
    parse_categorical_to_mask,
    parse_categorical_to_query,
    parse_inequality_to_mask,
    parse_inequality_to_query,
)
from .Types import Axis1D, Number

AxisLike: TypeAlias = SignalAxis | UnorderedSignalAxis | CategoricalAxis
Operand: TypeAlias = "DataBlock | Number | np.ndarray"
TwoSeriesOperation: TypeAlias = Callable[[pd.Series, pd.Series], pd.Series]


logger = logging.getLogger(__name__)


def return_axis_idx(axis: SignalAxis | UnorderedSignalAxis) -> int:
    return axis.index_in_array


def _selector_to_mask(axis: AxisLike, selector: object) -> np.ndarray:
    """Boolean mask over the points of `axis` for an indexing `selector`: either an
    expression string like 'x>2' or a 1-D boolean array/list/dask array with one
    entry per point."""
    if isinstance(selector, str):
        if isinstance(axis, CategoricalAxis):
            return parse_categorical_to_mask(axis.name, selector, axis.points)
        return parse_inequality_to_mask(axis.name, selector, axis.points)

    if not isinstance(selector, list | tuple | np.ndarray | da.Array):
        raise TypeError(
            f"Selector for axis '{axis.name}' must be an expression string or a boolean "
            f"array, got {type(selector)}"
        )
    mask = (
        selector.compute() if isinstance(selector, da.Array) else np.asarray(selector)
    )
    if mask.ndim != 1:
        raise ValueError(
            f"Mask for axis '{axis.name}' should be 1-D, got {mask.ndim} dimensions"
        )
    if mask.dtype != bool:
        raise TypeError(
            f"Mask for axis '{axis.name}' should be boolean, got dtype {mask.dtype}"
        )
    if len(mask) != axis.size:
        raise ValueError(
            f"Mask length ({len(mask)}) doesn't match axis '{axis.name}' size ({axis.size})"
        )
    return mask


def _filtered_axis(axis: AxisLike, mask: np.ndarray) -> AxisLike:
    """Copy of `axis` of the same type holding only the points where `mask` is True.
    Only a SignalAxis can be empty."""
    kept = np.asarray(axis.points)[mask]
    if len(kept) == 0 and not isinstance(axis, SignalAxis):
        raise ValueError(f"The selection on axis '{axis.name}' is empty")
    common = {
        "name": axis.name,
        "index_in_array": axis.index_in_array,
        "unit": axis.unit,
        "navigate": axis.is_nav,
    }
    if isinstance(axis, CategoricalAxis):
        return CategoricalAxis(kept.tolist(), **common)
    return type(axis)(kept, **common)


def compute_binning(
    axis: SignalAxis | UnorderedSignalAxis,
    bin_edges: Axis1D | None = None,
    bin_size: Number | None = None,
    eps: float = 1e-4,
) -> tuple[Axis1D, Axis1D]:
    if bin_edges is None:
        if bin_size is None:
            raise RuntimeError("Both 'new_binsize' and 'bin_edges' are None")

        num_bins = int(np.floor((axis.max - axis.min) / bin_size))
        bin_edges: Axis1D = np.linspace(axis.min - eps, axis.max + eps, num_bins + 1)
    else:
        # Work on a float copy: the padding below must not modify the caller's edges
        bin_edges = np.array(bin_edges, dtype=float)

    # 'Safety' padding has proven to be necessary
    bin_edges[0] -= eps
    bin_edges[-1] += eps
    bin_centers: Axis1D = bin_edges[1:] - (bin_edges[1:] - bin_edges[:-1]) / 2

    return bin_edges, bin_centers


def _normalise_drop_axes(drop_axes: list[str] | str | None) -> list[str]:
    """The axis names in `drop_axes` as a list; None means no axes."""
    if drop_axes is None:
        return []
    if isinstance(drop_axes, str):
        return [drop_axes]
    if isinstance(drop_axes, list):
        return drop_axes
    raise ValueError(
        f"drop_axes should be list[str], str or None.\nGot {type(drop_axes)}"
    )


def _pool_points(axis: AxisLike, window: int, remainder: str) -> tuple[np.ndarray, int]:
    """Points of `axis` cut (trim) or NaN-padded (pad) to whole windows, and the
    number of padded points. Raises ValueError where `remainder` forbids it."""
    leftover = axis.size % window
    points = np.asarray(axis.points)
    if leftover and remainder == "error":
        raise ValueError(
            f"Axis {axis.name} of size {axis.size} is not divisible by {window}"
        )
    if remainder == "trim":
        if axis.size < window:
            raise ValueError(
                f"Window {window} is larger than axis {axis.name} of size {axis.size}"
            )
        return points[: axis.size - leftover], 0
    if leftover:
        pad = window - leftover
        return np.pad(points.astype(float), (0, pad), constant_values=np.nan), pad
    return points, 0


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
        data: da.Array | np.ndarray,
        axes: Iterable[SignalAxis | CategoricalAxis],
        quantity: str = "q",
        unit: str = "-",
    ) -> None:
        axes_list: list = list(axes)
        if not len(data.shape) == len(axes_list):
            raise RuntimeError("Mismatched number of dimensions in data and axes")
        shape = data.shape

        axes: list[AxisLike] = sorted(axes, key=return_axis_idx)

        for i in range(len(axes)):
            axis: AxisLike = axes[i]
            if shape[i] != len(axis.points):
                raise RuntimeError(
                    f"Shape along dimension {i} has length {shape[i]} but axis has length {len(axis.points)}"
                )
            if shape[i] != axis.size:
                raise RuntimeError(
                    f"Shape along dimension {i} has length {shape[i]} but axis has size {axis.size}"
                )

        # If any axis has size 1 we squeeze the data. Squeezing a categorical axis
        # selects one category, so the quantity then becomes the category's axis name.
        if any(axis.size == 1 for axis in axes_list):
            squeeze_axes = [axis.index_in_array for axis in axes_list if axis.size == 1]
            data = da.squeeze(data, axis=tuple(squeeze_axes))
            for axis in axes_list:
                if axis.size == 1 and isinstance(axis, CategoricalAxis):
                    quantity = axis.name
            axes_list = [axis for axis in axes_list if axis.size > 1]
            for i, axis in enumerate(axes_list):
                axis.index_in_array = i

        self.data: da.Array | np.ndarray = data
        self.axes: list[SignalAxis | CategoricalAxis] = axes_list
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

    def __post_init_db__(self) -> None:  # ruff: ignore[bad-dunder-method-name]
        self.axes.sort(key=return_axis_idx)
        axis_indices: list[int] = [axis.index_in_array for axis in self.axes]
        axis_sizes: list[int] = [axis.size for axis in self.axes]
        data_shape: tuple[int, ...] = self.data.shape
        for dim in range(self.dims):
            if not axis_sizes[dim] == data_shape[dim]:
                raise RuntimeError(
                    f"Mismatch along dimension {dim}\n{self.axes[dim].name} has length {axis_sizes[dim]} but data has length {data_shape[dim]} along dimension"
                )
            if axis_indices[dim] != dim:
                logger.warning(
                    "fixed axis index, %s at dim=%d had index: %d",
                    self.axes[dim].name,
                    dim,
                    axis_indices[dim],
                )
                self.axes[dim].index_in_array = dim

        self.shape: tuple[int, ...] = tuple(axis.size for axis in self.axes)

    def rename_quantity(self, new_name: str) -> Self:
        new_self = deepcopy(self)
        new_self.quantity = new_name
        return new_self

    def c_op(self, axis: str, operation: Callable) -> Self:
        raise NotImplementedError()

    def _moved(
        self, axis_name: str, target_index: int
    ) -> tuple[list[AxisLike], da.Array | np.ndarray]:
        """Axes (deep copies) and data with `axis_name` moved to `target_index`,
        without modifying this datablock. Validates the arguments."""
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis: {axis_name}")
        if not -self.dims <= target_index < self.dims:
            raise ValueError(
                f"target_index {target_index} out of range for {self.dims} dimensions"
            )

        target_index %= self.dims
        old_index = self.axis_obj(axis_name).index_in_array

        new_axes = deepcopy(self.axes)
        new_axes.insert(target_index, new_axes.pop(old_index))
        for i, axis in enumerate(new_axes):
            axis.index_in_array = i

        array_module = np if isinstance(self.data, np.ndarray) else da
        return new_axes, array_module.moveaxis(self.data, old_index, target_index)

    def reorder_axis(self, axis_name: str, target_index: int) -> Self:
        """
        Returns a new datablock in which axis `axis_name` is moved to dimension
        `target_index`, shifting the axes in between, like `np.moveaxis`. Both axes
        and data are reordered; this datablock is not modified.
        See `reorder_axis_inplace` to reorder this datablock itself.

        Args:
            axis_name: str, name of axis to reorder
            target_index: int, dimensional position to move axis to, negative
                values count from the last dimension

        Returns:
            DataBlock; datablock with the reordered axes, same quantity and unit

        Raises:
            RuntimeError: If axis with `axis_name` not known.
            ValueError: If `target_index` is outside the dimensions of the datablock.
        """
        new_axes, new_data = self._moved(axis_name, target_index)
        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def reorder_axis_inplace(self, axis_name: str, target_index: int) -> None:
        """
        Moves axis `axis_name` to dimension `target_index` of this datablock itself,
        see `reorder_axis` for the semantics and a version that returns a new block.
        Will call `__post_init_db__` to verify data shape and axes match.

        Args:
            axis_name: str, name of axis to reorder
            target_index: int, dimensional position to move axis to, negative
                values count from the last dimension

        Raises:
            RuntimeError: If axis with `axis_name` not known.
            ValueError: If `target_index` is outside the dimensions of the datablock.
        """
        self.axes, self.data = self._moved(axis_name, target_index)
        self.__post_init_db__()

    def compute(self) -> np.ndarray:
        """Computes or returns the quantity in self.data depending on
        whether it has been computed before.

        Returns:
            np.ndarray; computed data
        """
        if not self._computed:
            self.data: da.Array = self.data.compute()
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
        axes_data: list[AxisLike] = []
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
                    axis.points.tolist()
                    if hasattr(axis.points, "tolist")
                    else list(axis.points)
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
    def load(cls, filepath: str | Path) -> Self:
        """Load DataBlock from disk. Data is always loaded lazily using dask.

        Args:
            filepath: str or Path; path to the saved DataBlock

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
        data = da.from_zarr(str(filepath / "data"))

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
        self,
        axis_name: str,
        min_idx: int | None = None,
        max_idx: int | None = None,
    ) -> Generator:
        """Yields the slices of the datablock along axis `axis_name`, in order.

        Args:
            axis_name: str; name of axis to iterate over
            min_idx: int | None; first index along the axis, defaults to the start
            max_idx: int | None; index along the axis to stop before, defaults to the end

        Yields:
            array of dimension `dims - 1`; the data at one index of the axis

        Raises:
            RuntimeError: If datablock does not have axis with name `axis_name`
        """
        if self.has_axis(axis_name):
            axis_idx = self.axis_obj(axis_name).index_in_array
            data = np.moveaxis(self.data, axis_idx, 0)[min_idx:max_idx]
            for i in range(data.shape[0]):
                yield data[i]
        else:
            raise RuntimeError(f"Axis {axis_name} not found in {self.axes}")

    def quantity_data(self) -> da.Array | np.ndarray:
        return self.data

    def has_axis(self, axis_name: str) -> bool:
        """returns True if datablock has axis with `axis_name`

        Args:
            axis_name: str;

        Returns:
            bool;
        """
        return any(axis.name == axis_name for axis in self.axes)

    def axis_obj(self, axis_name: str) -> SignalAxis | CategoricalAxis:
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

    def axis_p(self, axis_name: str) -> np.ndarray:
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

    def _binary_op(
        self,
        other: "DataBlock | Number | np.ndarray",
        operation: Callable,
        *,
        reflected: bool = False,
    ) -> Self:
        """Applies `operation(self, other)`, or `operation(other, self)` when
        `reflected`, elementwise. `other` is a scalar, an array that broadcasts
        against the data, or a DataBlock with identical axes. The result keeps the
        quantity and unit of this datablock and owns copies of its axes."""
        if isinstance(other, DataBlock):
            same_axes = len(self.axes) == len(other.axes) and all(
                mine.name == theirs.name
                and mine.size == theirs.size
                and np.array_equal(mine.points, theirs.points)
                for mine, theirs in zip(self.axes, other.axes, strict=True)
            )
            if not same_axes:
                raise ValueError(
                    "Operands should have identical axes (name, size and points)"
                )
            other = other.data
        left, right = (other, self.data) if reflected else (self.data, other)
        return type(self)(
            operation(left, right), deepcopy(self.axes), self.quantity, self.unit
        )

    def __add__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.add, reflected=True)

    def __sub__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.sub, reflected=True)

    def __mul__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.mul, reflected=True)

    def __truediv__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.truediv, reflected=True)

    def __pow__(self, other: Operand) -> Self:
        return self._binary_op(other, operator.pow)

    def __neg__(self) -> Self:
        return type(self)(-self.data, deepcopy(self.axes), self.quantity, self.unit)

    def crop_axis(self, axis_name: str, min: float, max: float) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"Axis {axis_name} not found")

        if isinstance(self.axis_obj(axis_name), CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        relevant_axis: SignalAxis | CategoricalAxis = self.axis_obj(axis_name)

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
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: np.ndarray[tuple[int,], np.dtype] | None = None,
        reducer: Callable = da.mean,
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
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        bin_edges: np.ndarray[tuple[int], np.dtype]
        bin_centers: np.ndarray[tuple[int], np.dtype]

        bin_edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        groups = np.digitize(relevant_axis.points, bins=bin_edges, right=True)

        seq = []
        new_axis_points = []
        for group in np.unique(groups):
            if group > len(bin_centers):
                continue  # Values are to the right of final bin_edge
            if group == 0:
                continue  # Values are to the left of first bin_edge
            mask = groups == group
            view = da.take(self.data, indices=mask, axis=relevant_axis.index_in_array)
            new_axis_points.append(bin_centers[group - 1])
            seq.append(reducer(view, axis=relevant_axis.index_in_array))
        new_view = da.stack(seq=seq, axis=relevant_axis.index_in_array)

        new_axis = SignalAxis(
            axis_points=np.array(new_axis_points),
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )
        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = new_axis
        return type(self)(new_view, new_axes, self.quantity, self.unit)

    def pool(
        self,
        windows: dict[str, int],
        reducer: Callable = np.mean,
        remainder: Literal["trim", "error", "pad"] = "trim",
    ) -> Self:
        """Pools blocks of neighbouring samples along one or more axes, for example
        `pool({"x": 3, "y": 3})` averages every 3x3 block of the `x` and `y` axes
        and leaves all other axes untouched. The operation stays lazy.

        The new points of a pooled axis are the mean of the points in each window.

        Args:
            windows: dict[str, int]; axis name -> number of samples per window
            reducer: Callable; reduction with numpy's `axis=` signature, applied
                to every window, by default the mean
            remainder: how to treat samples that do not fill a last window:
                "trim" drops them, "error" raises a ValueError and "pad" fills the
                last window with NaN (use a nan-aware `reducer` like `np.nanmean`).

        Returns:
            DataBlock; datablock with the pooled axes, same quantity and unit

        Raises:
            RuntimeError: If an axis is unknown or categorical.
            ValueError: If a window is not a positive integer, `remainder` is
                unknown, a window is larger than its axis while trimming, or the
                axis does not divide into windows while `remainder="error"`.
        """
        if remainder not in {"trim", "error", "pad"}:
            raise ValueError(
                f"remainder should be 'trim', 'error' or 'pad', got {remainder!r}"
            )

        factors = self._pool_factors(windows)
        data = da.asarray(self.data)
        new_axes = deepcopy(self.axes)
        window_mean = np.nanmean if remainder == "pad" else np.mean
        for index, window in factors.items():
            axis = self.axes[index]
            points, pad = _pool_points(axis, window, remainder)
            if pad:
                widths = [(0, pad) if i == index else (0, 0) for i in range(data.ndim)]
                data = da.pad(data.astype(float), widths, constant_values=np.nan)
            new_axes[index] = type(axis)(
                window_mean(points.reshape(-1, window), axis=1),
                name=axis.name,
                index_in_array=axis.index_in_array,
                unit=axis.unit,
                navigate=axis.is_nav,
            )

        if factors:
            data = da.coarsen(reducer, data, factors, trim_excess=remainder == "trim")
        if isinstance(self.data, np.ndarray):
            data = data.compute()
        return type(self)(data, new_axes, self.quantity, self.unit)

    def _pool_factors(self, windows: dict[str, int]) -> dict[int, int]:
        """Validated windows larger than 1, keyed by the index of their axis."""
        factors: dict[int, int] = {}
        for axis_name, window in windows.items():
            if not self.has_axis(axis_name):
                raise RuntimeError(f"Axis {axis_name} not found")
            axis = self.axis_obj(axis_name)
            if isinstance(axis, CategoricalAxis):
                raise RuntimeError(
                    f"Method not implemented for axis CategoricalAxis {axis_name}"
                )
            if isinstance(window, bool) or not isinstance(window, int | np.integer):
                raise ValueError(
                    f"Window for {axis_name} should be an int, got {window!r}"
                )
            if window < 1:
                raise ValueError(f"Window for {axis_name} should be >= 1, got {window}")
            if window > 1:
                factors[axis.index_in_array] = int(window)
        return factors

    def reduce_axis(
        self,
        axis_name: str | Sequence[str],
        reducer: Callable = lambda s: s.max(),
        out_quantity: str = "q",
        out_unit: str = "-",
    ) -> Self:
        """Fully reduces the axis, or axes, with name `axis_name`. Default reducer
        is .max(), works similarily to np.max(array, axis=n) but with named axes.

        `reducer` is applied with `dask.array.apply_along_axis` to every 1-D slice
        along the reduced axis, so it can be any function of a 1-D array (a median,
        a fit, ...) and the operation stays lazy. Dask rechunks the reduced axis into
        a single chunk for this, so one such slice column should fit in memory.

        When several axes are given they are flattened into one axis (in array
        order) before `reducer` is applied, so `reducer` sees all of their values
        at once. This matters for reducers that are not associative like a median.

        Args:
            axis_name: str | Sequence[str]; name(s) of the axis/axes to reduce
            reducer: Callable; function to reduce a 1-D array to a point
            out_quantity: str; name of quantity in returned datablock
            out_unit: str; name of the unit of quantity

        Returns:
            DataBlock; datablock without the reduced axes. A block of which all axes
                are reduced has no axes and a 0-d array as data.

        Raises:
            RuntimeError: If an axis is unknown.
            ValueError: If no axis is given or an axis is given twice.
        """
        names: list[str] = (
            [axis_name] if isinstance(axis_name, str) else list(axis_name)
        )
        if not names:
            raise ValueError("Give at least one axis to reduce")
        if len(set(names)) != len(names):
            raise ValueError(f"Axes to reduce should be unique, got {names}")
        for name in names:
            if not self.has_axis(name):
                raise RuntimeError(f"Axis {name} not found")

        reduced = sorted(self.axis_obj(name).index_in_array for name in names)
        kept = [i for i in range(self.dims) if i not in reduced]

        new_axes = [deepcopy(self.axes[i]) for i in kept]
        for i, axis in enumerate(new_axes):
            axis.index_in_array = i

        if len(reduced) == 1:
            new_data = da.apply_along_axis(
                func1d=reducer, axis=reduced[0], arr=self.data
            )
        else:
            data = da.moveaxis(da.asarray(self.data), reduced, range(-len(reduced), 0))
            data = data.reshape((*(self.shape[i] for i in kept), -1))
            new_data = da.apply_along_axis(func1d=reducer, axis=-1, arr=data)

        return type(self)(new_data, new_axes, quantity=out_quantity, unit=out_unit)

    def _select(self, indexing: dict) -> tuple[da.Array | np.ndarray, list[AxisLike]]:
        """Data and (new) axes after applying the selectors in `indexing`."""
        if not isinstance(indexing, dict):
            raise TypeError(f"Expected dict, got {type(indexing)}")

        new_data = self.data
        new_axes = deepcopy(self.axes)
        for axis_name, selector in indexing.items():
            if not self.has_axis(axis_name):
                raise ValueError(f"Axis '{axis_name}' not found in DataBlock")

            axis_obj = self.axis_obj(axis_name)
            mask = _selector_to_mask(axis_obj, selector)
            new_data = da.compress(
                condition=mask, a=new_data, axis=axis_obj.index_in_array
            )
            new_axes[axis_obj.index_in_array] = _filtered_axis(axis_obj, mask)
        return new_data, new_axes

    def get(self, indexing: dict) -> Self:
        """
        Advanced indexing method using dictionary specification.

        Args:
            indexing: Dictionary mapping axis names to either:
                     - Inequality string (numeric axes): 'x<3', 'x>2', '-3<=x<3',
                       'x==2', 'x!=2'; the axis name may be left out of a simple
                       comparison ('>2')
                     - Equality string (categorical axes): 'category==dog',
                       'label!=positive', "city=='new york'"
                     - Boolean array/list with one entry per point of the axis

        Axes not mentioned in the dict are not indexed (all values kept). The axis
        types are kept; selecting no point is only possible on a SignalAxis.

        Examples:
            datablock.get({'x': 'x>2', 'y': 'y<1', 'e': '-3<=e<3'})
            datablock.get({'category': 'category==dog'})
            datablock.get({'x': np.array([False, True, True, ...])})
            datablock.get({'x': 'x>0', 'y': [True, False, True]})

        Returns:
            DataBlock: New DataBlock with filtered data and updated axes

        Raises:
            TypeError: If `indexing` is no dict, or a selector is not a string or
                boolean array.
            ValueError: If an axis is unknown, an expression is invalid or a mask
                has the wrong length or dimension.
        """
        new_data, new_axes = self._select(indexing)
        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def value_get(
        self, indexing: dict
    ) -> tuple[da.Array | np.ndarray, dict[str, Axis1D]]:
        """
        Advanced indexing method that returns raw values instead of a DataBlock.
        Accepts the same `indexing` as `get`.

        Examples:
            quantity_data, axes_dict = datablock.value_get({'x': 'x>2', 'y': 'y<1'})
            # quantity_data is a dask array with the filtered data
            # axes_dict is {'x': np.array([...]), 'y': np.array([...]), ...}

        Returns:
            tuple: (quantity_data, axes_dict) where:
                - quantity_data: dask.array.Array with filtered quantity values
                - axes_dict: dict mapping axis names to their numpy arrays
        """
        new_data, new_axes = self._select(indexing)
        return new_data, {axis.name: axis.points for axis in new_axes}

    def cluster(self, num_clusters: int = 3) -> Self:
        """
        clusters the quantity data and returns a datablock with same axes and cluster label as data
        """

        if not DASKML:
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
        split_blocks: list[Self] = []
        split_axis: AxisLike = self.axis_obj(axis_name)
        if isinstance(split_axis, UnorderedSignalAxis):
            raise NotImplementedError("?? Unwanted")

        split_coords: np.ndarray | Axis1D = split_axis.points
        for val in split_coords:
            new_axis = CategoricalAxis(
                categories=[f"{val:.3f}"],
                name=split_axis.name,
                index_in_array=split_axis.index_in_array,
                unit=split_axis.unit,
                navigate=False,
            )
            new_axes: list[SignalAxis | CategoricalAxis] = deepcopy(self.axes)
            new_axes[new_axis.index_in_array] = new_axis
            new_data = da.take(
                self.data, indices=(split_coords == val), axis=new_axis.index_in_array
            )

            split_blocks.append(
                type(self)(
                    data=new_data, axes=new_axes, quantity=self.quantity, unit=self.unit
                )
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
        drop_names = _normalise_drop_axes(drop_axes)

        for axis in other.axes:
            if not self.has_axis(axis.name):
                raise RuntimeError(
                    f"Split on datablock has axes unkown to the to be splitted datablock\n\t{axis.name} not found for {self}"
                )

        if drop_names:
            ragged = [axis.name for axis in other.axes if axis.name not in drop_names]
            if ragged:
                raise ValueError(
                    "Every axis of `other` must be in drop_axes, otherwise the "
                    f"selected values do not form a regular block. Keeping: {ragged}"
                )

        missing: list[int] = [
            axis.index_in_array for axis in self.axes if not other.has_axis(axis.name)
        ]
        new_axes: list[AxisLike] = [
            deepcopy(axis) for axis in self.axes if axis.name not in drop_names
        ]
        for i, axis in enumerate(new_axes):
            axis.index_in_array = i
        new_shape = tuple(axis.size for axis in new_axes)

        split_blocks: list[Self] = []
        unique_values = np.sort(da.unique(other.quantity_data()).compute())
        for uval in unique_values:
            mask = other.quantity_data() == uval
            mask = da.expand_dims(mask, axis=tuple(missing[::-1]))
            mask = da.broadcast_to(mask, self.data.shape)

            if drop_names:
                new_data = self._collect_matching(mask, drop_names, new_shape)
                block_axes = [
                    *deepcopy(new_axes),
                    SignalAxis(
                        axis_points=np.arange(new_data.shape[-1]),
                        name="repetition",
                        index_in_array=len(new_axes),
                        unit="#",
                        navigate=False,
                    ),
                ]
            else:
                new_data = da.where(mask, self.data, da.ones(self.data.shape) * np.nan)
                block_axes = deepcopy(new_axes)
            split_blocks.append(
                type(self)(new_data, block_axes, self.quantity, self.unit)
            )
        return split_blocks

    def _collect_matching(
        self, mask: da.Array, drop_names: list[str], kept_shape: tuple[int, ...]
    ) -> da.Array:
        """Values where `mask` is True as an array of shape (*kept_shape, repetitions).

        The dropped axes are moved to the front first: the selected values then come
        out grouped per position along the kept axes, which the reshape relies on."""
        dropped_idx = [
            axis.index_in_array for axis in self.axes if axis.name in drop_names
        ]
        kept_idx = [
            axis.index_in_array for axis in self.axes if axis.name not in drop_names
        ]
        order = (*dropped_idx, *kept_idx)
        matching = da.transpose(self.data, order)[da.transpose(mask, order)]
        matching.compute_chunk_sizes()
        n_repetitions = matching.shape[0] // int(np.prod(kept_shape))
        return da.moveaxis(matching.reshape((n_repetitions, *kept_shape)), 0, -1)

    def to_ensemble(self) -> "Ensemble":
        return Ensemble.from_datablock(datablock=self)


class Ensemble:
    """
    Container for data that does not nicely conform to a coordinate grid
    """

    def __init__(
        self,
        data: dd.DataFrame | pd.DataFrame,
        axes: list[UnorderedSignalAxis | SignalAxis | CategoricalAxis],
        quantity: str = "q",
        unit: str = "-",
    ) -> None:
        self.data: dd.DataFrame | pd.DataFrame = data
        self.axes: list[UnorderedSignalAxis | SignalAxis | CategoricalAxis] = axes
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
                    axis.points.tolist()
                    if hasattr(axis.points, "tolist")
                    else list(axis.points)
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
    def load(cls, filepath: str | Path) -> Self:
        """Load Ensemble from disk. Data is always loaded lazily using dask.

        Args:
            filepath: str or Path; path to the saved Ensemble

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
        data = dd.read_parquet(str(parquet_path), engine="pyarrow")

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
    def from_datablock(cls, datablock: DataBlock) -> Self:
        data: da.Array | np.ndarray = datablock.data.flatten()
        axes: list[SignalAxis | CategoricalAxis] = datablock.axes
        quantity = datablock.quantity
        unit = datablock.unit
        mgrid = da.meshgrid(*[axis.points for axis in axes], indexing="ij")

        new_axes: list[SignalAxis | CategoricalAxis] = []
        for _, axis in enumerate(datablock.axes):
            new_axis = SignalAxis(
                axis_points=axis.points,
                name=axis.name,
                index_in_array=axis.index_in_array,
                unit=axis.unit,
                navigate=axis.is_nav,
            )
            if isinstance(axis, SignalAxis):
                new_axis.scale = axis.scale
            new_axes.append(new_axis)
        axes: list[SignalAxis | CategoricalAxis] = new_axes
        coord_block: da.Array = (
            da
            .stack([mgrid[i].flatten() for i in range(len(axes))] + [data])
            .rechunk({0: -1, 1: "auto"}, block_size_limit=1e8)
            .T
        )
        data: dd.DataFrame = dd.io.from_dask_array(
            coord_block,
            columns=[axis.name for axis in axes] + [quantity],
        )
        return Ensemble(data, axes, quantity, unit)  # type: ignore

    def __sub__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] -= other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __add__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] += other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __mul__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] *= other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __pow__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] **= other
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Enesmble trackin {self.quantity} across {len(self.data.columns) - 1} coordinates"

    def reorder_axis(self) -> None:
        raise NotImplementedError()

    def crop_axis(self) -> None:
        raise NotImplementedError()

    def has_axis(self, axis_name: str) -> bool:
        return any(axis.name == axis_name for axis in self.axes)

    def axis_obj(self, axis_name: str) -> AxisLike:
        if not self.has_axis(axis_name):
            raise ValueError(f"{self} has no axis: {axis_name}")
        for axis in self.axes:
            if axis.name == axis_name:
                return axis
        raise RuntimeError("Should not reach here")

    def axis_p(self, axis_name: str) -> Axis1D | None:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        for axis in self.axes:
            if axis.name == axis_name:
                return axis.points

        return None

    def cluster(self) -> None:
        raise NotImplementedError()

    def reduce_axis(
        self,
        axis_name: str | Sequence[str],
        reducer: dd.groupby.Aggregation,
        out_quantity: str = "q",
        out_unit: str = "-",
    ) -> Self:
        """Fully reduces the axis, or axes, with name `axis_name`: the quantity
        is aggregated over all rows that share their coordinates on the other axes.

        Args:
            axis_name: str | Sequence[str]; name(s) of the axis/axes to reduce
            reducer: aggregation understood by pandas/dask `groupby.agg`, for
                example "max", "sum" or a `dd.Aggregation`
            out_quantity: str; name of quantity in returned ensemble, the data
                column is renamed to match
            out_unit: str; name of the unit of quantity

        Returns:
            Ensemble; ensemble without the reduced axes

        Raises:
            RuntimeError: If an axis is unknown.
            ValueError: If no axis is given, an axis is given twice, all axes
                are reduced or `out_quantity` is the name of a remaining axis.
        """
        names: list[str] = (
            [axis_name] if isinstance(axis_name, str) else list(axis_name)
        )
        if not names:
            raise ValueError("Give at least one axis to reduce")
        if len(set(names)) != len(names):
            raise ValueError(f"Axes to reduce should be unique, got {names}")
        for name in names:
            if not self.has_axis(name):
                raise RuntimeError(f"Axis {name} not found")

        new_axes = [deepcopy(axis) for axis in self.axes if axis.name not in names]
        if not new_axes:
            raise ValueError("Cannot reduce all axes of an Ensemble, keep at least one")
        for i, axis in enumerate(new_axes):
            axis.index_in_array = i

        sort_axes = [axis.name for axis in new_axes]
        if out_quantity in sort_axes:
            raise ValueError(f"out_quantity '{out_quantity}' is the name of an axis")

        new_view = (
            self.data
            .groupby(sort_axes, observed=True)[self.quantity]
            .agg(reducer)
            .reset_index()
            .rename(columns={self.quantity: out_quantity})
        )

        return type(self)(new_view, new_axes, quantity=out_quantity, unit=out_unit)

    def rebin_reduce_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Axis1D | None = None,
        reducer: Callable = np.mean,
    ) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        relevant_axis: AxisLike = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        bin_edges: np.ndarray[tuple[int], np.dtype]
        bin_centers: np.ndarray[tuple[int], np.dtype]
        bin_edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        rebin_axis = relevant_axis.name + "_bin"
        sort_axes = [axis.name for axis in self.axes] + [rebin_axis]
        sort_axes.remove(relevant_axis.name)

        new_view = self.data.assign(**{
            rebin_axis: self.data[relevant_axis.name].map_partitions(
                pd.cut, bins=bin_edges, include_lowest=True, labels=False
            )
        })
        new_view = new_view.dropna()
        new_view = (
            new_view
            .groupby(sort_axes, observed=True)[self.quantity]
            .agg(list, meta=(self.quantity, "float64"))
            .reset_index()
        )

        new_view[self.quantity] = new_view[self.quantity].apply(
            reducer, meta=(relevant_axis.name, "float64")
        )

        mapping: dict[int, float] = dict(enumerate(bin_centers))
        new_view[relevant_axis.name] = new_view[rebin_axis].map(
            mapping, meta=(relevant_axis.name, "float64")
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

    def pool(self, bin_sizes: dict[str, Number], reducer: Callable = np.mean) -> Self:
        """Pools several axes at once, for example `pool({"x": 3.0, "y": 3.0})`.

        An Ensemble has no regular sample grid, so unlike `DataBlock.pool` the sizes
        are bin widths in axis units, and each named axis is rebinned with
        `rebin_reduce_axis`. Axes that are not named are untouched.

        Args:
            bin_sizes: dict[str, Number]; axis name -> new bin width
            reducer: Callable; function that reduces the values of one bin to a
                single value, by default the mean

        Returns:
            Ensemble; ensemble with the rebinned axes, same quantity and unit

        Raises:
            RuntimeError: If an axis is unknown or categorical.
        """
        result = type(self)(self.data, deepcopy(self.axes), self.quantity, self.unit)
        for axis_name, bin_size in bin_sizes.items():
            result = result.rebin_reduce_axis(
                axis_name, new_binsize=bin_size, reducer=reducer
            )
        return result

    def rebin_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Sequence[Number] | None = None,
    ) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        relevant_axis: AxisLike = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        bin_edges: np.ndarray[tuple[int], np.dtype]
        bin_centers: np.ndarray[tuple[int], np.dtype]
        bin_edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        rebin_axis = relevant_axis.name + "_bin"
        sort_axes = [axis.name for axis in self.axes] + [rebin_axis]
        sort_axes.remove(relevant_axis.name)

        if isinstance(self.data, dd.DataFrame):
            self.data: dd.DataFrame
            new_view = self.data.assign(**{
                rebin_axis: self.data[relevant_axis.name].map_partitions(
                    pd.cut, bins=bin_edges, include_lowest=True, labels=False
                )
            })
        else:
            self.data: pd.DataFrame
            new_view = self.data.assign(**{
                rebin_axis: pd.cut(
                    self.data[relevant_axis.name],
                    bins=bin_edges,
                    include_lowest=True,
                    labels=False,
                )
            })

        new_view = new_view.dropna()

        mapping: dict[int, float] = dict(enumerate(bin_centers))
        if isinstance(new_view, dd.DataFrame):
            new_view[relevant_axis.name]: dd.DataFrame = new_view[rebin_axis].map(
                mapping, meta=(relevant_axis.name, "float64")
            )
        else:
            new_view[relevant_axis.name]: pd.DataFrame = (
                new_view[rebin_axis].map(mapping).fillna(1_000_000)
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
        new_axes[relevant_axis.index_in_array].scale = new_binsize

        return type(self)(new_view, new_axes, self.quantity, self.unit)

    def rebin_to_new_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Axis1D | None = None,
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

        if new_binsize is None and bin_edges is None:
            raise RuntimeError("Either call with new_binsize= or bin_edges=")

        for axis in self.axes:
            if axis.name == axis_name:
                relevant_axis: AxisLike = axis

        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        bin_edges: np.ndarray[tuple[int], np.dtype]
        bin_centers: np.ndarray[tuple[int], np.dtype]
        bin_edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        rebin_axis = relevant_axis.name + "_bin"

        # Assign bin indices
        new_view = self.data.assign(**{
            rebin_axis: self.data[relevant_axis.name].map_partitions(
                pd.cut, bins=bin_edges, include_lowest=True, labels=False
            )
        }).dropna()

        mapping = dict(enumerate(bin_centers))

        new_view[relevant_axis.name] = new_view[rebin_axis].map(
            mapping, meta=(relevant_axis.name, "float64")
        )

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

    def to_datablock(self, rebin_scheme: dict[str, Number] | None = None) -> DataBlock:
        if rebin_scheme is None:
            rebin_scheme = {}
        for axis in self.axes:
            if not isinstance(axis, SignalAxis):
                raise RuntimeError(
                    f"can only make datablock of monotonic axes\nAxis: {axis.name} not monotonic"
                )

        quantity_data: pd.Series | dd.Series = self.data[self.quantity]
        if isinstance(quantity_data, dd.Series):
            values: da.Array = quantity_data.to_dask_array(lengths=True).rechunk()
        else:
            values: da.Array = da.array(quantity_data)

        coords = da.stack(
            [
                ((self.data[axis.name] - axis.min) / axis.scale)
                .astype(np.int64)
                .to_dask_array(lengths=True)
                for axis in self.axes
            ],
            axis=0,
        ).rechunk((-1, values.chunksize[0]))
        nshape: tuple[int, ...] = tuple(ax.size for ax in self.axes)

        data_db = da.map_blocks(
            lambda c, v: sparse.COO(c, v, shape=nshape, fill_value=np.nan),
            coords,
            values,
            dtype=values.dtype,
        ).map_blocks(
            lambda b: b.todense(),
            dtype=values.dtype,
        )
        data_db = data_db.compute()  # TODO: does this work without?

        return DataBlock(data_db, deepcopy(self.axes), self.quantity, self.unit)

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
        if other.quantity not in self.data.columns:
            raise RuntimeError(
                f"Can not perform operation, self.data has not column {other.quantity}"
            )

        self_data: dd.DataFrame = self.data
        other_data: dd.DataFrame = other.data

        left_axes = [ax.name for ax in self.axes]
        right_axes = [ax.name for ax in other.axes]
        common_axes = list(set(left_axes) & set(right_axes))

        def _function(
            partition: pd.DataFrame,
            other_data: pd.DataFrame,
            operation: TwoSeriesOperation,
        ) -> pd.DataFrame:
            new_part = partition.merge(
                other_data, how="left", on=common_axes, suffixes=("", "_other")
            )
            new_part[other.quantity] = operation(
                new_part[other.quantity], new_part[other.quantity + "_other"]
            )
            new_part.drop(columns=other.quantity + "_other")
            return new_part

        new_data = self_data.map_partitions(_function, other_data, operation)

        if other.quantity in left_axes:
            old_axis: SignalAxis | UnorderedSignalAxis = self.axes[
                left_axes.index(other.quantity)
            ]
            new_axis = UnorderedSignalAxis(
                axis_points=new_data[other.quantity],
                name=old_axis.name,
                index_in_array=old_axis.index_in_array,
                unit=old_axis.unit,
                navigate=old_axis.is_nav,
            )
            new_axes = [
                new_axis if ax.name == other.quantity else ax for ax in self.axes
            ]
        else:
            new_axes = deepcopy(self.axes)

        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def combine(self, other: Self) -> Self:
        self_data: pd.DataFrame = self.data
        other_data: pd.DataFrame = other.data

        left_axes = [ax.name for ax in self.axes]
        right_axes = [ax.name for ax in other.axes]
        common_axes = list(set(left_axes) & set(right_axes))

        new_data = self_data.merge(
            other_data, how="left", on=common_axes, suffixes=("", "_other")
        )
        old_axes = self.axes
        new_axis = UnorderedSignalAxis(
            axis_points=other.data[other.quantity],
            name=other.quantity,
            index_in_array=len(self.axes),
            unit=other.unit,
            navigate=False,
        )
        return type(self)(new_data, [*old_axes, new_axis], self.quantity, self.unit)

    def _select(self, indexing: dict) -> dd.DataFrame | pd.DataFrame:
        """Rows of the data that satisfy all selectors in `indexing`."""
        if not isinstance(indexing, dict):
            raise TypeError(f"Expected dict, got {type(indexing)}")

        new_data = self.data
        for axis_name, selector in indexing.items():
            if not self.has_axis(axis_name):
                raise ValueError(f"Axis '{axis_name}' not found in Ensemble")

            axis_obj = self.axis_obj(axis_name)
            if isinstance(selector, str):
                if isinstance(axis_obj, CategoricalAxis):
                    query_str = parse_categorical_to_query(axis_name, selector)
                else:
                    query_str = parse_inequality_to_query(axis_name, selector)
                new_data = new_data.query(query_str)
            else:
                # A boolean mask selects points of the axis, like for a DataBlock
                mask = _selector_to_mask(axis_obj, selector)
                keep_values = np.asarray(axis_obj.points)[mask]
                new_data = new_data[new_data[axis_name].isin(keep_values)]
        return new_data

    def get(self, indexing: dict) -> Self:
        """
        Advanced indexing method using dictionary specification.

        Args:
            indexing: Dictionary mapping axis names to either:
                     - Inequality string (numeric axes): 'x<3', 'x>2', '-3<=x<3',
                       'x==2', 'x!=2'; the axis name may be left out of a simple
                       comparison ('>2')
                     - Equality string (categorical axes): 'category==dog',
                       'label!=positive', "city=='new york'"
                     - Boolean array/list with one entry per point of the axis

        Axes not mentioned in the dict are not indexed (all values kept). The
        filtered axes keep the points that remain, in their original order.

        Examples:
            ensemble.get({'x': 'x>2', 'y': 'y<1', 'e': '-3<=e<3'})
            ensemble.get({'category': 'category==dog'})
            ensemble.get({'x': np.array([True, False, True, ...])})
            ensemble.get({'x': 'x>0', 'y': [True, False, True]})

        Returns:
            Ensemble: New Ensemble with filtered data

        Raises:
            TypeError: If `indexing` is no dict, or a selector is not a string or
                boolean array.
            ValueError: If an axis is unknown, an expression is invalid or a mask
                has the wrong length or dimension.
        """
        new_data = self._select(indexing)

        new_axes: list[AxisLike] = []
        for axis in self.axes:
            if axis.name in indexing:
                present = new_data[axis.name].unique()
                if hasattr(present, "compute"):
                    present = present.compute()
                remaining = np.isin(np.asarray(axis.points), np.asarray(present))
                new_axes.append(_filtered_axis(axis, remaining))
            else:
                new_axes.append(deepcopy(axis))

        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def value_get(
        self, indexing: dict
    ) -> tuple[np.ndarray | da.Array, dict[str, np.ndarray]]:
        """
        Advanced indexing method that returns raw values instead of an Ensemble.
        Accepts the same `indexing` as `get`; boolean masks select points of the
        axis in both.

        Examples:
            quantity_values, axes_dict = ensemble.value_get({'x': 'x>2', 'y': 'y<1'})
            # quantity_values is a numpy/dask array with the filtered quantity values
            # axes_dict is {'x': np.array([...]), 'y': np.array([...]), ...}

        Returns:
            tuple: (quantity_values, axes_dict) where:
                - quantity_values: numpy/dask array with filtered quantity values
                - axes_dict: dict mapping axis names to their values per row
        """
        new_data = self._select(indexing)

        def values(column: dd.Series | pd.Series) -> np.ndarray | da.Array:
            # Text columns can hold pandas string extension dtypes, which dask only
            # converts to arrays with a warning
            if column.dtype.kind not in "biufc":
                column = column.astype(object)
            return column.values

        axes_dict = {axis.name: values(new_data[axis.name]) for axis in self.axes}
        return values(new_data[self.quantity]), axes_dict

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

        split_ensembles: list[Self] = []
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
                    data=filtered_data,
                    axes=new_axes,
                    quantity=self.quantity,
                    unit=self.unit,
                )
            )

        return split_ensembles

    def _axes_present_in(
        self, data: dd.DataFrame | pd.DataFrame, drop_axes: list[str]
    ) -> list[AxisLike]:
        """Copies of the axes not in `drop_axes`, keeping the points found in `data`."""
        new_axes: list[AxisLike] = []
        for axis in self.axes:
            if axis.name in drop_axes:
                continue
            column = data[axis.name]
            if hasattr(column, "compute"):
                column = column.compute()
            present = np.isin(np.asarray(axis.points), column.unique())
            new_axes.append(_filtered_axis(axis, present))
            new_axes[-1].index_in_array = len(new_axes) - 1
        return new_axes

    def split_with_ensemble(
        self, other: Self, drop_axes: list[str] | str
    ) -> list[Self]:
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

        split_ensembles: list[Self] = []

        # Get unique mask values; rows without a match in `other` (NaN) are excluded
        unique_values = merged_data[other.quantity].dropna().unique()
        if hasattr(unique_values, "compute"):
            unique_values = unique_values.compute()
        unique_values = np.sort(np.asarray(unique_values))

        for uval in unique_values:
            # Filter merged data for this unique value
            filtered_data = merged_data[merged_data[other.quantity] == uval]

            # Every split gets its own axes holding only the coordinates it contains
            new_axes = self._axes_present_in(filtered_data, drop_axes)

            # Drop the mask column and any dropped axes
            columns_to_keep = [ax.name for ax in new_axes] + [self.quantity]
            filtered_data = filtered_data[columns_to_keep]

            split_ensembles.append(
                type(self)(
                    filtered_data, new_axes, quantity=self.quantity, unit=self.unit
                )
            )

        return split_ensembles
