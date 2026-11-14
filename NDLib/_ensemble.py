"""The Ensemble class: a collection of DataBlocks sharing metadata."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import (
    Self,
    cast,
)

import dask.array as da
import dask.dataframe as dd
from dask.dataframe.groupby import Aggregation
import numpy as np
import pandas as pd
import sparse
from typing_extensions import override
import zarr

from ._datablock import DataBlock
from ._helpers import (
    AxisLike,
    TwoSeriesOperation,
    _axis_from_record,
    _axis_to_record,
    _filtered_axis,
    _Metadata,
    _read_metadata,
    _selector_to_mask,
    compute_binning,
)
from .AxesStructures import (
    CategoricalAxis,
    SignalAxis,
    UnorderedSignalAxis,
)
from .IndexingUtils import (
    parse_categorical_to_query,
    parse_inequality_to_query,
)
from .Types import Axis1D, Number

logger = logging.getLogger(__name__)


class Ensemble:
    """
    Container for data that does not nicely conform to a coordinate grid
    """

    def __init__(
        self,
        data: dd.DataFrame | pd.DataFrame,
        axes: list[AxisLike],
        quantity: str = "q",
        unit: str = "-",
    ) -> None:
        super().__init__()
        self.data: dd.DataFrame | pd.DataFrame = data
        self.axes: list[AxisLike] = axes
        self.dims: int = len(axes)
        self.quantity: str = quantity
        self.unit: str = unit
        if isinstance(self.data, dd.DataFrame):
            self._computed: bool = False
        else:
            self._computed = True

    def compute(self) -> pd.DataFrame:
        if isinstance(self.data, dd.DataFrame):
            self.data = self.data.compute()
        self._computed = True
        return cast("pd.DataFrame", self.data)

    def _dask_data(self) -> dd.DataFrame:
        """The data as a lazy dask DataFrame. After `compute` the data is a pandas
        DataFrame, which is wrapped in a single partition again so that the lazy
        operations keep working. `self.data` itself is not modified."""
        if isinstance(self.data, dd.DataFrame):
            return self.data
        return dd.from_pandas(self.data, npartitions=1)

    def _merge_inputs(
        self, other: Self
    ) -> tuple[dd.DataFrame | pd.DataFrame, dd.DataFrame | pd.DataFrame]:
        """Data of `self` and `other` that can be merged with each other: both
        pandas stays pandas, otherwise both become lazy dask DataFrames."""
        if isinstance(self.data, pd.DataFrame) and isinstance(other.data, pd.DataFrame):
            return self.data, other.data
        return self._dask_data(), other._dask_data()

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

        # Save metadata as JSON
        metadata = _Metadata(
            quantity=self.quantity,
            unit=self.unit,
            axes=[_axis_to_record(axis) for axis in self.axes],
        )
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
        metadata = _read_metadata(root)

        # Load dataframe
        parquet_path = filepath / "dataframe.parquet"
        data = dd.read_parquet(str(parquet_path), engine="pyarrow")

        return cls(
            data=data,
            axes=[_axis_from_record(record) for record in metadata["axes"]],
            quantity=metadata["quantity"],
            unit=metadata["unit"],
        )

    @classmethod
    def from_datablock(cls, datablock: DataBlock) -> Self:
        values = da.asarray(datablock.data).ravel()
        mgrid = list(
            da.meshgrid(*[axis.points for axis in datablock.axes], indexing="ij")
        )

        new_axes: list[AxisLike] = []
        for axis in datablock.axes:
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
        coord_block: da.Array = (
            da
            .stack([grid.ravel() for grid in mgrid] + [values])
            .rechunk({0: -1, 1: "auto"}, block_size_limit=1e8)
            .T
        )
        data = dd.from_dask_array(
            coord_block,
            columns=[axis.name for axis in new_axes] + [datablock.quantity],
        )
        return cls(data, new_axes, datablock.quantity, datablock.unit)

    def __sub__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] -= cast("float", other)
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __add__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] += cast("float", other)
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __mul__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] *= cast("float", other)
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    def __pow__(self, other: Number) -> Self:
        new_data = self.data.copy()
        new_data[self.quantity] **= cast("float", other)
        return type(self)(new_data, self.axes, self.quantity, self.unit)

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
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
        reducer: Aggregation | str | Callable[..., object],
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

        grouped = self.data.groupby(sort_axes, observed=True)[self.quantity]
        aggregated = cast(
            "pd.Series",
            grouped.agg(reducer),  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue, reportArgumentType]
        )
        new_view = aggregated.reset_index().rename(
            columns={self.quantity: out_quantity}
        )

        return type(self)(new_view, new_axes, quantity=out_quantity, unit=out_unit)

    def rebin_reduce_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Axis1D | Sequence[Number] | None = None,
        reducer: Callable[..., object] = np.mean,
    ) -> Self:
        if not self.has_axis(axis_name):
            raise RuntimeError(f"{self} has no axis {axis_name}")

        relevant_axis = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        rebin_axis = relevant_axis.name + "_bin"
        sort_axes = [axis.name for axis in self.axes] + [rebin_axis]
        sort_axes.remove(relevant_axis.name)

        data = self._dask_data()
        new_view = data.assign(**{
            rebin_axis: data[relevant_axis.name].map_partitions(
                pd.cut, bins=edges, include_lowest=True, labels=False
            )
        })
        new_view = new_view.dropna()
        new_view = (
            new_view
            .groupby(sort_axes, observed=True)[self.quantity]
            .agg(list, meta=(self.quantity, "float64"))
            .reset_index()
        )

        # dask calls the reducer on the list of values of every group
        group_reducer = cast("Callable[..., pd.Series]", reducer)
        new_view[self.quantity] = new_view[self.quantity].apply(
            group_reducer, meta=(relevant_axis.name, "float64")
        )

        mapping: dict[int, float] = dict(enumerate(bin_centers))
        new_view[relevant_axis.name] = new_view[rebin_axis].map(
            mapping.get, meta=(relevant_axis.name, "float64")
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

    def pool(
        self, bin_sizes: dict[str, Number], reducer: Callable[..., object] = np.mean
    ) -> Self:
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

        relevant_axis = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        rebin_axis = relevant_axis.name + "_bin"
        sort_axes = [axis.name for axis in self.axes] + [rebin_axis]
        sort_axes.remove(relevant_axis.name)

        if isinstance(self.data, dd.DataFrame):
            new_view = self.data.assign(**{
                rebin_axis: self.data[relevant_axis.name].map_partitions(
                    pd.cut, bins=edges, include_lowest=True, labels=False
                )
            })
        else:
            new_view = self.data.assign(**{
                rebin_axis: pd.cut(
                    self.data[relevant_axis.name],
                    bins=edges.tolist(),
                    include_lowest=True,
                    labels=False,
                )
            })

        new_view = new_view.dropna()

        mapping: dict[int, float] = dict(enumerate(bin_centers))
        if isinstance(new_view, dd.DataFrame):
            new_view[relevant_axis.name] = new_view[rebin_axis].map(
                mapping.get, meta=(relevant_axis.name, "float64")
            )
        else:
            new_view[relevant_axis.name] = new_view[rebin_axis].map(mapping.get)

        new_view = new_view.drop(columns=rebin_axis)

        new_axes = deepcopy(self.axes)
        new_axes[relevant_axis.index_in_array] = SignalAxis(
            axis_points=bin_centers,
            name=relevant_axis.name,
            index_in_array=relevant_axis.index_in_array,
            unit=relevant_axis.unit,
            navigate=relevant_axis.is_nav,
        )
        if new_binsize is not None:
            new_axes[relevant_axis.index_in_array].scale = new_binsize

        return type(self)(new_view, new_axes, self.quantity, self.unit)

    def rebin_to_new_axis(
        self,
        axis_name: str,
        new_binsize: Number | None = None,
        bin_edges: Axis1D | Sequence[Number] | None = None,
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

        relevant_axis = self.axis_obj(axis_name)
        if isinstance(relevant_axis, CategoricalAxis):
            raise RuntimeError(
                f"Method not implemented for axis CategoricalAxis {axis_name}"
            )

        edges, bin_centers = compute_binning(relevant_axis, bin_edges, new_binsize)

        rebin_axis = relevant_axis.name + "_bin"

        # Assign bin indices
        data = self._dask_data()
        new_view = data.assign(**{
            rebin_axis: data[relevant_axis.name].map_partitions(
                pd.cut, bins=edges, include_lowest=True, labels=False
            )
        }).dropna()

        mapping = dict(enumerate(bin_centers))

        new_view[relevant_axis.name] = new_view[rebin_axis].map(
            mapping.get, meta=(relevant_axis.name, "float64")
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

        data = self._dask_data()
        values = data[self.quantity].to_dask_array(lengths=True).rechunk()

        def grid_index(axis: AxisLike) -> da.Array:
            # position of every row along the regular grid of `axis`
            column = cast("dd.Series", data[axis.name])
            steps = cast(
                "dd.Series",
                (column - cast("float", axis.min)) / cast("float", axis.scale),  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
            )
            return steps.astype(np.int64).to_dask_array(lengths=True)

        coords = da.stack([grid_index(axis) for axis in self.axes], axis=0).rechunk((
            -1,
            max(values.chunks[0]),
        ))
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

        self_data = self._dask_data()
        other_data = other._dask_data()

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
            old_axis = self.axes[left_axes.index(other.quantity)]
            new_axis = UnorderedSignalAxis(
                axis_points=cast("Axis1D", new_data[other.quantity]),
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
        self_data, other_data = self._merge_inputs(other)

        left_axes = [ax.name for ax in self.axes]
        right_axes = [ax.name for ax in other.axes]
        common_axes = list(set(left_axes) & set(right_axes))

        new_data = cast("pd.DataFrame", self_data).merge(
            cast("pd.DataFrame", other_data),
            how="left",
            on=common_axes,
            suffixes=("", "_other"),
        )
        old_axes = self.axes
        new_axis = UnorderedSignalAxis(
            axis_points=cast("Axis1D", other.data[other.quantity]),
            name=other.quantity,
            index_in_array=len(self.axes),
            unit=other.unit,
            navigate=False,
        )
        return type(self)(new_data, [*old_axes, new_axis], self.quantity, self.unit)

    def _select(self, indexing: Mapping[str, object]) -> dd.DataFrame | pd.DataFrame:
        """Rows of the data that satisfy all selectors in `indexing`."""
        if not isinstance(indexing, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
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
                column = new_data[axis.name]
                present = (
                    column.unique().compute()
                    if isinstance(column, dd.Series)
                    else column.unique()
                )
                remaining = np.isin(np.asarray(axis.points), np.asarray(present))
                new_axes.append(_filtered_axis(axis, remaining))
            else:
                new_axes.append(deepcopy(axis))

        return type(self)(new_data, new_axes, self.quantity, self.unit)

    def value_get(
        self, indexing: Mapping[str, object]
    ) -> tuple[np.ndarray | da.Array, dict[str, np.ndarray | da.Array]]:
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
            if isinstance(column, dd.Series):
                return column.values
            return np.asarray(column.values)

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
        split_axis = self.axis_obj(axis_name)
        split_values = np.asarray(split_axis.points)

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
        self_data, other_data = self._merge_inputs(other)

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
