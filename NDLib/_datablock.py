"""The DataBlock class: an n-dimensional array with typed, labelled axes."""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
import importlib.util
import json
import logging
import operator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypeAlias,
    cast,
)

import dask.array as da
import numpy as np
from typing_extensions import override
import zarr

from . import Selection
from .AxesStructures import (
    CategoricalAxis,
    SignalAxis,
    UnorderedSignalAxis,
)
from .Types import Axis1D, Number

if TYPE_CHECKING:
    from ._ensemble import Ensemble

from ._helpers import (
    AxisLike,
    LabelKey,
    _axis_from_points,
    _axis_from_record,
    _axis_to_record,
    _describe,
    _filtered_axis,
    _json_default,
    _Metadata,
    _moveaxis,
    _normalise_drop_axes,
    _pool_points,
    _read_metadata,
    _selector_to_mask,
    compute_binning,
    return_axis_idx,
)

# dask-ml is optional, it is imported when DataBlock.cluster needs it
DASKML: bool = importlib.util.find_spec("dask_ml") is not None

Operand: TypeAlias = "DataBlock | Number | np.ndarray"

logger = logging.getLogger(__name__)


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
        metadata: dict; interpreted, calibrated acquisition metadata (JSON-able)
        original_metadata: dict; metadata exactly as found in the source file (shared
            with derived blocks, treat as read-only)
        provenance: list[str]; human-readable record of the operations applied

    Axes flagged `is_nav` are navigation axes (the scan, e.g. x and y of a spectrum
    image); the others are signal axes (the data recorded at each navigation point,
    e.g. the energy loss). See `nav_axes`, `signal_axes` and `with_navigation`.
    Pass `squeeze=False` to keep axes of size 1 instead of squeezing them away.
    """

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        data: da.Array | np.ndarray,
        axes: Iterable[AxisLike],
        quantity: str = "q",
        unit: str = "-",
        *,
        metadata: Mapping[str, Any] | None = None,
        original_metadata: Mapping[str, Any] | None = None,
        provenance: Iterable[str] | None = None,
        squeeze: bool = True,
    ) -> None:
        super().__init__()
        axes_list: list[AxisLike] = list(axes)
        if not len(data.shape) == len(axes_list):
            raise RuntimeError("Mismatched number of dimensions in data and axes")
        shape = data.shape

        sorted_axes = sorted(axes_list, key=return_axis_idx)

        for i, axis in enumerate(sorted_axes):
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
        if squeeze and any(axis.size == 1 for axis in axes_list):
            squeeze_axes = [axis.index_in_array for axis in axes_list if axis.size == 1]
            data = da.squeeze(data, axis=tuple(squeeze_axes))
            for axis in axes_list:
                if axis.size == 1 and isinstance(axis, CategoricalAxis):
                    quantity = axis.name
            axes_list = [axis for axis in axes_list if axis.size > 1]
            for i, axis in enumerate(axes_list):
                axis.index_in_array = i

        self.data: da.Array | np.ndarray = data
        self.axes: list[AxisLike] = axes_list
        self.dims: int = len(self.data.shape)
        self.quantity: str = quantity
        self.unit: str = unit
        self.metadata: dict[str, Any] = deepcopy(dict(metadata or {}))
        self.original_metadata: dict[str, Any] = deepcopy(dict(original_metadata or {}))
        self.provenance: list[str] = list(provenance or [])
        if isinstance(self.data, da.Array):
            self._computed: bool = False
        else:
            self._computed = True
        self.__post_init_db__()

    @override
    def __repr__(self) -> str:
        return f"{self.__class__}\n"

    @override
    def __str__(self) -> str:
        return self.__repr__()

    def __post_init_db__(self) -> None:  # ruff: ignore[bad-dunder-method-name]
        self.axes.sort(key=return_axis_idx)
        axis_indices: list[int] = [axis.index_in_array for axis in self.axes]
        axis_sizes: list[int] = [axis.size for axis in self.axes]
        data_shape = self.data.shape
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

    def _construct(
        self,
        data: da.Array | np.ndarray,
        axes: list[AxisLike],
        quantity: str,
        unit: str,
        *,
        squeeze: bool = True,
    ) -> Self:
        """Create a block of this class from `data` and `axes`.

        This is the single place where derived blocks are instantiated. A subclass
        whose `__init__` takes more or different arguments overrides it to pass them
        on (it should also accept `squeeze`). Without an override a subclass that
        defines its own `__init__` gets a plain `DataBlock` when `squeeze` is False,
        as its constructor arguments are not known here.
        """
        if squeeze:
            return type(self)(data, axes, quantity, unit)
        if type(self).__init__ is DataBlock.__init__:
            return type(self)(data, axes, quantity, unit, squeeze=False)
        return cast("Self", DataBlock(data, axes, quantity, unit, squeeze=False))

    def _inherit(self, new: Self, step: str | None = None) -> Self:
        """Copy this block's metadata and provenance onto `new` and return it,
        optionally appending a provenance `step`. Subclasses extend this to carry
        their own state over to derived blocks.

        `metadata` is copied. `original_metadata`, the file's own record (often
        thousands of tags), is shared between a block and the blocks derived from it
        so that deriving stays cheap: treat it as read-only.
        """
        new.metadata = deepcopy(self.metadata)
        new.original_metadata = self.original_metadata
        new.provenance = [*self.provenance, *([step] if step else [])]
        return new

    def _derive(  # ruff: ignore[too-many-arguments]
        self,
        data: da.Array | np.ndarray,
        axes: list[AxisLike],
        *,
        quantity: str | None = None,
        unit: str | None = None,
        step: str | None = None,
        squeeze: bool = True,
    ) -> Self:
        """The block derived from this one: `_construct` followed by `_inherit`.

        `quantity` and `unit` default to this block's. Every operation that returns a
        new block goes through here, so subclasses only override `_construct` and
        `_inherit` to survive all of them.
        """
        new = self._construct(
            data,
            axes,
            self.quantity if quantity is None else quantity,
            self.unit if unit is None else unit,
            squeeze=squeeze,
        )
        return self._inherit(new, step)

    @property
    def nav_axes(self) -> list[AxisLike]:
        """The navigation axes (`is_nav`), in array order."""
        return [axis for axis in self.axes if axis.is_nav]

    @property
    def signal_axes(self) -> list[AxisLike]:
        """The signal axes (not `is_nav`), in array order."""
        return [axis for axis in self.axes if not axis.is_nav]

    @property
    def nav_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.nav_axes)

    @property
    def signal_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.signal_axes)

    def with_navigation(self, *axis_names: str) -> Self:
        """New block in which exactly the named axes are navigation axes."""
        unknown = [name for name in axis_names if not self.has_axis(name)]
        if unknown:
            raise KeyError(f"No such axes: {unknown}")
        new_axes = deepcopy(self.axes)
        for axis in new_axes:
            axis.is_nav = axis.name in axis_names
        return self._derive(self.data, new_axes, step=f"with_navigation{axis_names}")

    def isel(self, **indexers: int | slice) -> Self:
        """Select by integer position along named axes, keeping the calibration.

        An int drops the axis, a slice keeps it (sub-ranged, a `LinearAxis` stays
        linear). Selection is lazy and cheap for dask data. Axes of size 1 are kept.
        A subclass with its own `__init__` keeps its type if it overrides `_construct`.

        Example: `sb.isel(x=3, y=slice(0, 10))`
        """
        resolved, new_axes = Selection.plan(
            self.axes, indexers, Selection.resolve_position
        )
        data = Selection.apply(self.data, resolved)
        return self._derive(
            cast("da.Array | np.ndarray", data),
            cast("list[AxisLike]", new_axes),
            step=f"isel({_describe(indexers)})",
            squeeze=False,
        )

    def sel(self, **indexers: LabelKey) -> Self:
        """Select by calibrated coordinate along named axes, keeping the calibration.

        A number selects the nearest point and drops the axis; `slice(lo, hi)`
        keeps points with lo <= value <= hi (either bound may be None); on a
        categorical axis a name or a list of names. Axes of size 1 are kept.

        Example: `si.sel(x=slice(2.0, 4.0), E=slice(280.0, 300.0))`
        """
        resolved, new_axes = Selection.plan(
            self.axes, indexers, Selection.resolve_label
        )
        data = Selection.apply(self.data, resolved)
        return self._derive(
            cast("da.Array | np.ndarray", data),
            cast("list[AxisLike]", new_axes),
            step=f"sel({_describe(indexers)})",
            squeeze=False,
        )

    def _reduced(self, axis_names: tuple[str, ...], how: str, *, skipna: bool) -> Self:
        if not axis_names:
            raise ValueError("Name at least one axis to reduce")
        unknown = [name for name in axis_names if not self.has_axis(name)]
        if unknown:
            raise KeyError(f"No such axes: {unknown}")
        positions = tuple(
            sorted({self.axis_obj(name).index_in_array for name in axis_names})
        )
        name = ("nan" if skipna else "") + how
        data = getattr(da if isinstance(self.data, da.Array) else np, name)(
            self.data, axis=positions
        )
        kept = [
            deepcopy(axis) for axis in self.axes if axis.index_in_array not in positions
        ]
        for i, axis in enumerate(kept):
            axis.index_in_array = i
        return self._derive(data, kept, step=f"{how}_over{axis_names}", squeeze=False)

    def sum_over(self, *axis_names: str, skipna: bool = False) -> Self:
        """Sum over the named axes, which are dropped. Uses dask's native reduction."""
        return self._reduced(axis_names, "sum", skipna=skipna)

    def mean_over(self, *axis_names: str, skipna: bool = False) -> Self:
        """Mean over the named axes, which are dropped. Uses dask's native reduction."""
        return self._reduced(axis_names, "mean", skipna=skipna)

    def rename_quantity(self, new_name: str) -> Self:
        new_self = deepcopy(self)
        new_self.quantity = new_name
        return new_self

    def c_op(self, axis: str, operation: Callable[..., object]) -> Self:
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

        return new_axes, _moveaxis(self.data, old_index, target_index)

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
        return self._derive(new_data, new_axes)

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
        if isinstance(self.data, da.Array):
            self.data = np.asarray(self.data.compute())
        self._computed = True
        return np.asarray(self.data)

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
            da.to_zarr(self.data, data_path, overwrite=True)
        else:
            zarr.save_array(data_path, self.data, overwrite=True)  # pyright: ignore[reportArgumentType]

        # Save metadata as JSON in the zarr group
        metadata = _Metadata(
            quantity=self.quantity,
            unit=self.unit,
            axes=[_axis_to_record(axis) for axis in self.axes],
        )
        if self.metadata:
            metadata["metadata"] = self.metadata
        if self.original_metadata:
            metadata["original_metadata"] = self.original_metadata
        if self.provenance:
            metadata["provenance"] = self.provenance
        root = zarr.open_group(str(filepath), mode="a")
        root.attrs["metadata"] = json.dumps(metadata, default=_json_default)

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
        metadata = _read_metadata(root)

        # Load data
        data = da.from_zarr(str(filepath / "data"))

        return cls(
            data=data,
            axes=[_axis_from_record(record) for record in metadata["axes"]],
            quantity=metadata["quantity"],
            unit=metadata["unit"],
            metadata=metadata.get("metadata"),
            original_metadata=metadata.get("original_metadata"),
            provenance=metadata.get("provenance"),
        )

    def iter_axis(
        self,
        axis_name: str,
        min_idx: int | None = None,
        max_idx: int | None = None,
    ) -> Iterator[da.Array | np.ndarray]:
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
            data = _moveaxis(self.data, axis_idx, 0)[min_idx:max_idx]
            for i in range(int(data.shape[0])):
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
        other: Operand,
        operation: Callable[..., da.Array | np.ndarray],
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
            operand: object = other.data
        else:
            operand = other
        left, right = (operand, self.data) if reflected else (self.data, operand)
        return self._derive(operation(left, right), deepcopy(self.axes))

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
        return self._derive(-self.data, deepcopy(self.axes))

    def crop_axis(self, axis_name: str, min: float, max: float) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"Axis {axis_name} not found")

        relevant_axis = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

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
        return self._derive(keep_values, new_axes)

    def rebin_reduce_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Axis1D | Sequence[Number] | None = None,
        reducer: Callable[..., object] = da.mean,
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
        relevant_axis = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        groups = np.digitize(relevant_axis.points, bins=edges, right=True)

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
        return self._derive(new_view, new_axes)

    def pool(
        self,
        windows: dict[str, int],
        reducer: Callable[..., object] = np.mean,
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
            # categorical axes were rejected in _pool_factors
            axis = cast("SignalAxis | UnorderedSignalAxis", self.axes[index])
            points, pad = _pool_points(axis, window, remainder)
            if pad:
                widths = [(0, pad) if i == index else (0, 0) for i in range(data.ndim)]
                data = da.pad(data.astype(float), widths, constant_values=np.nan)
            new_axes[index] = _axis_from_points(
                axis, window_mean(points.reshape(-1, window), axis=1)
            )

        if factors:
            data = da.coarsen(reducer, data, factors, trim_excess=remainder == "trim")
        if isinstance(self.data, np.ndarray):
            data = data.compute()
        return self._derive(data, new_axes)

    def _pool_factors(self, windows: Mapping[str, object]) -> dict[int, int]:
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
        reducer: Callable[..., object] = np.max,
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

        return self._derive(new_data, new_axes, quantity=out_quantity, unit=out_unit)

    def _select(
        self, indexing: Mapping[str, object]
    ) -> tuple[da.Array | np.ndarray, list[AxisLike]]:
        """Data and (new) axes after applying the selectors in `indexing`."""
        if not isinstance(indexing, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
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

    def get(self, indexing: Mapping[str, object]) -> Self:
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
        return self._derive(new_data, new_axes)

    def value_get(
        self, indexing: Mapping[str, object]
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

        dask_ml_cluster = importlib.import_module("dask_ml.cluster")
        clusterer = dask_ml_cluster.KMeans(n_clusters=num_clusters)
        data_reshaped = da.asarray(self.data).ravel().reshape((-1, 1))
        cluster_labels = clusterer.fit(data_reshaped).predict(data_reshaped)
        cluster_data = cluster_labels.reshape(self.data.shape)
        new_axes = deepcopy(self.axes)

        return self._derive(cluster_data, new_axes, quantity="cluster", unit="-")

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
            new_axes: list[AxisLike] = deepcopy(self.axes)
            new_axes[new_axis.index_in_array] = new_axis
            new_data = da.take(
                self.data, indices=(split_coords == val), axis=new_axis.index_in_array
            )

            split_blocks.append(self._derive(new_data, new_axes))
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
                    f"Every axis of `other` must be in drop_axes, otherwise the selected values do not form a regular block. Keeping: {ragged}"
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
        unique_values = np.sort(np.asarray(da.unique(other.quantity_data()).compute()))
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
            split_blocks.append(self._derive(new_data, block_axes))
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
        from ._ensemble import Ensemble  # ruff: ignore[import-outside-top-level] - Ensemble imports this module

        return Ensemble.from_datablock(datablock=self)
