# pyright: basic
from typing import Literal

import numpy as np

from .Types import Axis1D, Number


class Axis:
    """Axis base class

    Attributes:
        name: str; name of axis, should be unique in Ensemble or DataBlock
        index_in_array: int; axis index of corresponding datablock
        unit: str; name of axis unit
        is_nav: bool; unused as of now
        ordered: bool; whether axis monotonically increasing, currently not used
        size: int; number of points of the axis
        min: Number; minimum value of axis
        max: Number; maximum value of axis
        scale: Number; spacing of axis points
    """

    def __init__(
        self, name: str, index_in_array: int, unit: str, *, navigate: bool = False
    ) -> None:
        self.name: str = name
        self.index_in_array: int = index_in_array
        self.unit: str = unit
        self.is_nav: bool = navigate
        self.ordered: bool | None = None
        self.size: int = 0
        self.min: Number | None = None
        self.max: Number | None = None
        self.scale: Number | None = None

    def __repr__(self) -> str:
        return f"{self.__class__}  {self.name}  {self.unit}  {self.index_in_array}"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other: object, /):
        """Compares two axes by value, anything else elementwise against the points.

        Two axes are equal when they have the same type, name, unit and points; the
        position in the array and the navigation flag are layout and not compared.
        Against a scalar or array the result is the array of elementwise comparisons
        (subclasses decide whether that is supported, see `_compare_points`).
        """
        if isinstance(other, Axis):
            return (
                type(self) is type(other)
                and self.name == other.name
                and self.unit == other.unit
                and np.array_equal(self._points_array(), other._points_array())
            )
        return self._compare_points(other)

    def __hash__(self) -> int:
        """Hash consistent with `__eq__`. Do not modify `points` while an axis is
        stored in a set or used as a dictionary key."""
        points = self._points_array()
        if points.dtype.kind in "biuf":
            # int and float axes with equal values are equal, so hash them alike
            content: object = (points.astype(np.float64) + 0.0).tobytes()
        else:
            content = tuple(points.tolist())
        return hash((type(self).__name__, self.name, self.unit, content))

    def _points_array(self) -> np.ndarray:
        return np.asarray(getattr(self, "points", ()))

    def _compare_points(self, value: object) -> object:
        return NotImplemented


class SignalAxis(Axis):
    """Axis of strictly increasing points

    Raises ValueError for points that are not 1-D or not strictly increasing
    (unsorted, descending, duplicated or NaN); use an UnorderedSignalAxis for those.

    Attributes:
        points: np.ndarray | dask.array.Array; axis coordinates
        name: str; name of axis, should be unique in Ensemble or DataBlock
        index_in_array: int; axis index of corresponding datablock
        unit: str; name of axis unit
        is_nav: bool; unused as of now
        ordered: bool; whether axis monotonically increasing, currently not used
        size: int; number of points of the axis
        min: Number; minimum value of axis
        max: Number; maximum value of axis
        scale: Number; spacing of axis points
    """

    def __init__(
        self,
        axis_points: Axis1D,
        name: str,
        index_in_array: int,
        unit: str,
        *,
        navigate: bool = False,
    ) -> None:
        super().__init__(name, index_in_array, unit, navigate=navigate)

        self.points: Axis1D = np.array(axis_points)
        if self.points.ndim != 1:
            raise ValueError(
                f"Points of axis '{name}' should be 1-D, got {self.points.ndim} dimensions"
            )
        if self.points.size > 1 and not np.all(np.diff(self.points) > 0):
            raise ValueError(
                f"Points of SignalAxis '{name}' should be strictly increasing, "
                "use an UnorderedSignalAxis for other coordinates"
            )
        self.ordered = True
        self.size: int = len(self.points)
        if self.size > 1:
            self.scale: Number = self.points[1] - self.points[0]
            self.min: Number = self.points.min()
            self.max: Number = self.points.max()
        else:
            self.scale = np.inf

    def _compare_points(
        self, value: object
    ) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        return np.equal(self.points, value)

    def __gt__(self, value: object, /) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        return self.points > value

    def __ge__(self, value: object, /) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        return self.points >= value

    def __le__(self, value: object, /) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        return self.points <= value

    def __lt__(self, value: object, /) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        return self.points < value

    def __repr__(self) -> str:
        return f"Axis: {self.name}"

    def __str__(self) -> str:
        return (
            f"monotonic axis: '{self.name}'\t: {self.size} bins\n"
            + f"{self.min:.2f} {self.unit} <--------step: {self.scale:.2f} {self.unit}--------> {self.max:.2f} {self.unit}"
        )

class UnorderedSignalAxis(Axis):
    """[TODO:description]

    Attributes:
        points: np.ndarray | dask.array.Array; axis coordinates
        binned: bool; True when axis has monotonically increasing coordinates
        name: str; name of axis, should be unique in Ensemble or DataBlock
        index_in_array: int; axis index of corresponding datablock
        unit: str; name of axis unit
        is_nav: bool; unused as of now
        ordered: bool; whether axis monotonically increasing, currently not used
        size: int; number of points of the axis
        min: Number; minimum value of axis
        max: Number; maximum value of axis
        scale: Number; spacing of axis points
    """

    def __init__(
        self,
        axis_points: Axis1D,
        name: str,
        index_in_array: int,
        unit: str,
        *,
        navigate: bool = False,
    ) -> None:
        super().__init__(name, index_in_array, unit, navigate=navigate)

        self.points: Axis1D = axis_points
        self.binned: bool = False
        self.scale: Number | None = None
        self.ordered = False
        self.__post_init__()

    def __post_init__(self) -> None:
        self.size: int = self.points.size
        self.min: Number = self.points.min()
        self.max: Number = self.points.max()

    def __repr__(self) -> str:
        return f"{self.__class__}  {self.name}  {self.unit}  {self.index_in_array}"

    def __str__(self) -> str:
        return (
            "navigational " * self.is_nav
            + f"irregularly spaced {self.name} axis: {self.size} bins\n"
            + f"{self.min:.2f} {self.unit} <--------{self.size} steps--------> {self.min:.2f} {self.unit}"
        )

    def crop(self, min: Number | None = None, max: Number | None = None) -> None:
        if (min is None) & (max is None):
            raise RuntimeError("Both min and max are None")

        if min is not None and min > self.min:
            self.points = self.points[self.points > min]
            self.__post_init__()

        if max is not None and max < self.max:
            self.points = self.points[self.points < max]
            self.__post_init__()


class CategoricalAxis(Axis):
    """Axis of categories (strings)
        instead of x0 = 0, x1 = 1,... xn = n etc
        axis of ['center', 'amplitude', 'fwhm']

    Attributes:
        points: np.nndarray[str]; array of strings
        size: int; length of points
        ordered: bool; unused
    """

    def __init__(
        self,
        categories: list[str],
        name: str,
        index_in_array: int,
        unit: str,
        *,
        navigate: bool = False,
    ) -> None:
        super().__init__(name, index_in_array, unit, navigate=navigate)

        if categories == []:
            raise RuntimeError("Empty category list")

        self.points: np.ndarray = np.array(categories)
        self.size: int = len(self.points)
        self.ordered = False

    def __repr__(self) -> str:
        return f"{self.__class__}  {self.name}  {self.unit}  {self.index_in_array}"

    def __str__(self) -> str:
        return (
            f"categorical axis: {self.name}: {self.size} categories\n"
            + f"Categories: {', '.join(self.points)}"
        )

    def _compare_points(
        self, value: object
    ) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        points: np.ndarray[Literal[1], np.dtype[Number | np.str_]] = self.points
        return np.equal(points, value)
