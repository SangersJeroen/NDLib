# pyright: basic
from typing import Literal
import numpy as np
from .Types import Number, Axis1D
from typing import Optional


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

    def __init__(self, name: str, index_in_array: int, unit: str, navigate: bool):
        self.name: str = name
        self.index_in_array: int = index_in_array
        self.unit: str = unit
        self.is_nav: bool = navigate
        self.ordered: Optional[bool] = None
        self.size: int = 0
        self.min: Optional[Number] = None
        self.max: Optional[Number] = None
        self.scale: Optional[Number] = None

    def __repr__(self) -> str:
        return f"{self.__class__}  {self.name}  {self.unit}  {self.index_in_array}"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self.size


class SignalAxis(Axis):
    """Axis of monotonically increasing points

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
        self, axis_points: Axis1D, name: str, index_in_array: int, unit: str, navigate: bool
    ):
        super().__init__(name, index_in_array, unit, navigate)

        if len(axis_points) > 1:
            first_diff = axis_points[1] - axis_points[0]
            # assert np.allclose(
            #     axis_points[1:] - axis_points[:-1], first_diff
            # ), "Axis points are not monotonically increasing, try UnorderedSignalAxis"

        self.points: Axis1D = np.array(axis_points)
        self.ordered = True
        self.size: int = len(self.points)
        if self.size > 1:
            self.scale: Number = self.points[1] - self.points[0]
            self.min: Number = self.points.min()
            self.max: Number = self.points.max()
        else:
            self.scale = np.inf

    def __eq__(self, value: object, /) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
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
        self, axis_points: Axis1D, name: str, index_in_array: int, unit: str, navigate: bool
    ):
        super().__init__(name, index_in_array, unit, navigate)
        assert len(axis_points) > 1, "Axis is of length 1 or less"

        self.points: Axis1D = axis_points
        self.binned: bool = False
        self.scale: Optional[Number] = None
        self.ordered = False
        self.__post_init__()

    def __post_init__(self):
        self.size: int = len(self.points)
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

    def crop(self, min: Optional[Number] = None, max: Optional[Number] = None) -> None:
        assert (min is None) & (max is None), "Both min and max are None"

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
        self, categories: list[str], name: str, index_in_array: int, unit: str, navigate: bool
    ):
        super().__init__(name, index_in_array, unit, navigate)

        assert len(categories) > 0, "Categories list is empty"

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

    def __eq__(self, value: str, /) -> np.ndarray[Literal[1], np.dtype[np.bool_]]:
        points: np.ndarray[Literal[1], np.dtype[Number | np.str_]] = self.points
        return np.equal(points, value)
