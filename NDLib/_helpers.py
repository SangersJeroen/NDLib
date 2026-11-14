"""Private helpers shared by DataBlock and Ensemble (axis selection, serialisation records, binning)."""

from collections.abc import Callable, Mapping, Sequence
import json
import logging
from pathlib import Path
from typing import (
    Any,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

import dask.array as da
import numpy as np
import pandas as pd
import zarr

from .AxesStructures import (
    CategoricalAxis,
    LinearAxis,
    SignalAxis,
    UnorderedSignalAxis,
)
from .IndexingUtils import (
    parse_categorical_to_mask,
    parse_inequality_to_mask,
)
from .Types import Axis1D, Number

LabelKey: TypeAlias = "float | str | slice | Sequence[str]"
AxisLike: TypeAlias = (
    SignalAxis | UnorderedSignalAxis | CategoricalAxis
)  # LinearAxis is a SignalAxis
TwoSeriesOperation: TypeAlias = Callable[[pd.Series, pd.Series], pd.Series]


logger = logging.getLogger(__name__)


def _describe(indexers: Mapping[str, object]) -> str:
    return ", ".join(f"{name}={key!r}" for name, key in indexers.items())


def return_axis_idx(axis: AxisLike) -> int:
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
            f"Selector for axis '{axis.name}' must be an expression string or a boolean array, got {type(selector)}"
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
    if isinstance(axis, CategoricalAxis):
        return CategoricalAxis(
            kept.tolist(),
            axis.name,
            axis.index_in_array,
            axis.unit,
            navigate=axis.is_nav,
        )
    return _axis_from_points(axis, kept)


def _axis_from_points(axis: AxisLike, points: np.ndarray) -> AxisLike:
    """New axis of the kind of `axis` with the given `points`. A LinearAxis becomes
    a SignalAxis, as the new points need not be uniform."""
    axis_class = SignalAxis if isinstance(axis, LinearAxis) else type(axis)
    return axis_class(
        cast("Any", points),
        axis.name,
        axis.index_in_array,
        axis.unit,
        navigate=axis.is_nav,
    )


class _AxisRecord(TypedDict):
    """An axis as stored in the JSON metadata of a saved DataBlock or Ensemble."""

    type: str
    name: str
    index_in_array: int
    unit: str
    navigate: bool
    points: NotRequired[list[float]]
    categories: NotRequired[list[str]]
    offset: NotRequired[float]
    scale: NotRequired[float]
    size: NotRequired[int]


class _Metadata(TypedDict):
    quantity: str
    unit: str
    axes: list[_AxisRecord]
    metadata: NotRequired[dict[str, Any]]
    original_metadata: NotRequired[dict[str, Any]]
    provenance: NotRequired[list[str]]


def _json_default(value: object) -> object:
    """JSON encoder for the numpy values that commonly end up in metadata."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Metadata value of type {type(value)} is not JSON serializable")


def _axis_to_record(axis: AxisLike) -> _AxisRecord:
    record = _AxisRecord(
        type=type(axis).__name__,
        name=axis.name,
        index_in_array=axis.index_in_array,
        unit=axis.unit,
        navigate=axis.is_nav,
    )
    if isinstance(axis, LinearAxis):
        record["offset"] = axis.offset
        record["scale"] = axis.scale
        record["size"] = axis.size
        return record
    values = np.asarray(axis.points).tolist()
    if isinstance(axis, CategoricalAxis):
        record["categories"] = values
    else:
        record["points"] = values
    return record


def _axis_from_record(record: _AxisRecord) -> AxisLike:
    kind = record["type"]
    name = record["name"]
    index = record["index_in_array"]
    unit = record["unit"]
    navigate = record["navigate"]
    if kind == "CategoricalAxis":
        categories = record.get("categories")
        if categories is None:
            raise ValueError(f"Saved axis '{name}' has no categories")
        return CategoricalAxis(categories, name, index, unit, navigate=navigate)
    if kind == "LinearAxis":
        try:
            offset, scale, size = record["offset"], record["scale"], record["size"]
        except KeyError as error:
            raise ValueError(f"Saved axis '{name}' lacks {error}") from error
        return LinearAxis(offset, scale, size, name, index, unit, navigate=navigate)
    if kind not in {"SignalAxis", "UnorderedSignalAxis"}:
        raise ValueError(f"Unknown axis type: {kind}")
    points = record.get("points")
    if points is None:
        raise ValueError(f"Saved axis '{name}' has no points")
    axis_class = SignalAxis if kind == "SignalAxis" else UnorderedSignalAxis
    return axis_class(np.array(points), name, index, unit, navigate=navigate)


def _read_metadata(root: zarr.Group) -> _Metadata:
    return cast("_Metadata", json.loads(str(root.attrs["metadata"])))


def compute_binning(
    axis: SignalAxis | UnorderedSignalAxis,
    bin_edges: Axis1D | Sequence[Number] | None = None,
    bin_size: Number | None = None,
    eps: float = 1e-4,
) -> tuple[Axis1D, Axis1D]:
    if bin_edges is None:
        if bin_size is None:
            raise RuntimeError("Both 'new_binsize' and 'bin_edges' are None")

        num_bins = int(np.floor((axis.max - axis.min) / bin_size))  # type: ignore[operator]
        bin_edges = np.linspace(axis.min - eps, axis.max + eps, num_bins + 1)
    else:
        # Work on a float copy: the padding below must not modify the caller's edges
        bin_edges = np.array(bin_edges, dtype=float)

    # 'Safety' padding has proven to be necessary
    bin_edges[0] -= eps
    bin_edges[-1] += eps
    bin_centers: Axis1D = bin_edges[1:] - (bin_edges[1:] - bin_edges[:-1]) / 2

    return bin_edges, bin_centers


def _moveaxis(
    data: da.Array | np.ndarray, source: int, destination: int
) -> da.Array | np.ndarray:
    """`moveaxis` that keeps numpy arrays numpy and dask arrays dask."""
    if isinstance(data, np.ndarray):
        return np.moveaxis(data, source, destination)
    return da.moveaxis(data, source, destination)


def _normalise_drop_axes(drop_axes: object) -> list[str]:
    """The axis names in `drop_axes` as a list; None means no axes."""
    if drop_axes is None:
        return []
    if isinstance(drop_axes, str):
        return [drop_axes]
    if isinstance(drop_axes, list):
        return [str(name) for name in drop_axes]
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
