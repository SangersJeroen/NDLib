# pyright: basic
"""Calibration-preserving selection along axes.

`isel` selects by integer position, `sel` by calibrated coordinate. Both resolve a
per-axis key to either an integer (the axis is dropped), a slice, or an index array,
and rebuild the axis accordingly, keeping a `LinearAxis` linear when possible.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

import numpy as np

from .AxesStructures import (
    Axis,
    CategoricalAxis,
    LinearAxis,
    SignalAxis,
    UnorderedSignalAxis,
)

# a resolved key: int drops the axis, slice/array keeps it
Resolved: TypeAlias = int | slice | np.ndarray


def _empty(axis: Axis) -> ValueError:
    return ValueError(f"The selection on axis '{axis.name}' is empty")


def resolve_position(axis: Axis, key: object) -> Resolved:
    """Resolve an integer-position key (`isel`) for `axis`."""
    if isinstance(key, bool):
        raise TypeError(f"Invalid position for axis '{axis.name}': {key!r}")
    if isinstance(key, int | np.integer):
        index = int(key)
        if not -axis.size <= index < axis.size:
            raise IndexError(
                f"Index {index} is out of range for axis '{axis.name}' of size {axis.size}"
            )
        return index % axis.size
    if isinstance(key, slice):
        if (key.step or 1) < 1:
            raise ValueError(f"Slice step on axis '{axis.name}' must be positive")
        if len(range(*key.indices(axis.size))) == 0:
            raise _empty(axis)
        return key
    raise TypeError(
        f"isel key for axis '{axis.name}' must be an int or a slice, got {type(key)}"
    )


def resolve_label(axis: Axis, key: object) -> Resolved:
    """Resolve a calibrated-coordinate key (`sel`) for `axis`.

    A number selects the nearest point (and drops the axis). A slice selects the
    points between its bounds, both inclusive, either of which may be None. For a
    categorical axis a string selects that category and a list of strings those
    categories.
    """
    if isinstance(axis, CategoricalAxis):
        return _resolve_category(axis, key)

    if isinstance(key, slice):
        return _resolve_label_slice(axis, key)
    if isinstance(key, int | float | np.integer | np.floating) and not isinstance(
        key, bool
    ):
        if isinstance(axis, LinearAxis):
            return axis.index_of(float(key))
        return int(np.argmin(np.abs(np.asarray(axis.points) - key)))
    raise TypeError(
        f"sel key for axis '{axis.name}' must be a number or a slice, got {type(key)}"
    )


def _resolve_label_slice(axis: Axis, key: slice) -> Resolved:
    """Points of `axis` between the bounds of `key` (inclusive, None is open)."""
    if key.step is not None:
        raise ValueError(f"sel on axis '{axis.name}' does not support a step")
    points = np.asarray(axis.points)
    if isinstance(axis, SignalAxis):  # sorted points: bisect, giving a cheap slice
        start = (
            0 if key.start is None else int(np.searchsorted(points, key.start, "left"))
        )
        stop = (
            axis.size
            if key.stop is None
            else int(np.searchsorted(points, key.stop, "right"))
        )
        if stop <= start:
            raise _empty(axis)
        return slice(start, stop)
    mask = np.ones(axis.size, dtype=bool)
    if key.start is not None:
        mask &= points >= key.start
    if key.stop is not None:
        mask &= points <= key.stop
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise _empty(axis)
    return indices


def _resolve_category(axis: CategoricalAxis, key: object) -> Resolved:
    names = [str(p) for p in axis.points]
    if isinstance(key, str):
        if key not in names:
            raise KeyError(f"Category '{key}' not on axis '{axis.name}'")
        return names.index(key)
    if isinstance(key, list | tuple) and key and all(isinstance(k, str) for k in key):
        missing = [k for k in key if k not in names]
        if missing:
            raise KeyError(f"Categories {missing} not on axis '{axis.name}'")
        return np.array([names.index(k) for k in key])
    raise TypeError(
        f"sel key for categorical axis '{axis.name}' must be a string or a list of strings"
    )


def select_axis(axis: Axis, resolved: slice | np.ndarray) -> Axis:
    """A copy of `axis` restricted to `resolved`, same role, unit and position."""
    common = {"navigate": axis.is_nav}
    name, index, unit = axis.name, axis.index_in_array, axis.unit
    if isinstance(axis, CategoricalAxis):
        return CategoricalAxis(
            np.asarray(axis.points)[resolved].tolist(), name, index, unit, **common
        )
    if isinstance(axis, LinearAxis) and isinstance(resolved, slice):
        start, stop, step = resolved.indices(axis.size)
        return LinearAxis(
            axis.offset + axis.scale * start,
            axis.scale * step,
            len(range(start, stop, step)),
            name,
            index,
            unit,
            **common,
        )
    points = np.asarray(axis.points)[resolved]
    if isinstance(axis, UnorderedSignalAxis):
        return UnorderedSignalAxis(points, name, index, unit, **common)
    return SignalAxis(points, name, index, unit, **common)


def plan(
    axes: Sequence[Axis],
    keys: Mapping[str, object],
    resolver: Callable[[Axis, object], Resolved],
) -> tuple[dict[int, Resolved], list[Axis]]:
    """Resolve `keys` (by axis name) and build the axes of the result.

    Returns the resolved key per axis position and the new axes, renumbered; axes
    selected with an integer are dropped.
    """
    names = [axis.name for axis in axes]
    unknown = [name for name in keys if name not in names]
    if unknown:
        raise KeyError(f"No such axes: {unknown}; available: {names}")

    resolved: dict[int, Resolved] = {}
    new_axes: list[Axis] = []
    for axis in axes:
        if axis.name not in keys:
            new_axes.append(select_axis(axis, slice(None)))
            continue
        result = resolver(axis, keys[axis.name])
        resolved[axis.index_in_array] = result
        if not isinstance(result, int):
            new_axes.append(select_axis(axis, result))
    for i, axis in enumerate(new_axes):
        axis.index_in_array = i
    return resolved, new_axes


def apply(data: object, resolved: Mapping[int, Resolved]) -> object:
    """Index `data` along each resolved axis in turn (orthogonally).

    Axes are processed from the last to the first so that dropping one does not
    shift the positions of those still to do.
    """
    for position in sorted(resolved, reverse=True):
        key = resolved[position]
        data = data[(slice(None),) * position + (key,)]  # type: ignore[index]
    return data
