# NDLib: guide for LLMs and new contributors

NDLib (v0.1.0, split out of the private project SSEELS) processes n-dimensional
datasets whose dimensions are **named, labelled axes** (like xarray, but small and
dask-first). Python >= 3.11. Everything is lazy through dask unless noted.

Core dependencies: numpy, pandas, dask (array + dataframe), pyarrow, sparse, zarr.
`dask-ml` is optional and only needed by `DataBlock.cluster`.

## Public API

```python
from NDLib import DataBlock, Ensemble, SignalAxis, UnorderedSignalAxis
# CategoricalAxis is NOT exported at top level:
from NDLib.AxesStructures import CategoricalAxis
```

| Object | What it is | Backing data |
|---|---|---|
| `SignalAxis` | Strictly increasing numeric points (ValueError otherwise: unsorted, descending, duplicates, NaN, not 1-D) | `np.ndarray` |
| `UnorderedSignalAxis` | Numeric points in any order or with repeats | `np.ndarray` |
| `CategoricalAxis` | String labels, e.g. `["center", "amplitude", "fwhm"]` (empty list raises RuntimeError) | `np.ndarray[str]` |
| `DataBlock` | Dense, regular n-d grid: one value per axis-point combination | `dask.array` or `np.ndarray` |
| `Ensemble` | Irregular/sparse samples in "long" form: one row per sample, one column per axis plus the quantity | `dask.dataframe` or `pandas.DataFrame` |

Axis constructor: `SignalAxis(axis_points, name, index_in_array, unit, *, navigate=False)`.
`CategoricalAxis(categories, name, index_in_array, unit, *, navigate=False)`.
Axis attributes: `name, index_in_array, unit, is_nav, ordered, size, min, max, scale, points`.
Axes compare by value (type, name, unit, points; not position) and are hashable.
`axis > 3`, `axis == "dog"` etc. return elementwise boolean arrays (ordering only on `SignalAxis`).

## Quick start

```python
import numpy as np
from NDLib import DataBlock, SignalAxis
from NDLib.AxesStructures import CategoricalAxis

x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
db = DataBlock(np.random.rand(10, 5), [x, y], quantity="intensity", unit="counts")

sub = db.get({"x": "x>2", "y": "-1<=y<3"})        # filter by expression
mx  = db.reduce_axis("y", reducer=np.max, out_quantity="peak", out_unit="counts")
low = db.pool({"x": 2, "y": 5})                   # 2x5 block mean
rb  = db.rebin_reduce_axis("x", new_binsize=3.0)  # rebin to ~3 um bins
arr = (db * 2 + 1).compute()                      # np.ndarray

db.save("run1")                                   # writes run1.zarr
db2 = DataBlock.load("run1")                      # lazy dask

ens = db.to_ensemble()                            # long-form DataFrame view
back = ens.to_datablock()                         # needs regular SignalAxis grid
```

## Invariants that bite

- **Axis `name`s are unique** within a block and are the handle for every operation.
- **`index_in_array` must equal the axis position in the data.** `DataBlock.__init__`
  sorts axes by it; mismatches in `__post_init_db__` are auto-fixed with a logged warning.
- **Size-1 axes are squeezed away** on `DataBlock` construction. If the squeezed axis is
  categorical, the block's `quantity` becomes that axis's name. Remaining axes are renumbered.
- **Almost everything returns a new object** (deep-copied axes, `type(self)(...)`).
  Exceptions: `reorder_axis_inplace`, `compute()`, `Axis.crop` on `UnorderedSignalAxis`.
- **`compute()` materialises in place**: replaces `self.data` with numpy/pandas and
  returns it.
- Binning pads edges by `eps=1e-4` (`compute_binning`) and returns bin centres as the
  new axis points.
- Pytest runs with `-W error`: any warning fails a test.
- Typos preserved in the code, don't "fix" them when matching strings: `Enesmble` in `Ensemble.__str__`.

## DataBlock methods (`NDLib/BaseStructures.py:225`)

Introspection: `has_axis(name)`, `axis_obj(name)`, `axis_p(name)` (points),
`quantity_data()`, attrs `data, axes, dims, shape, quantity, unit`.

| Method | Behaviour |
|---|---|
| `get(indexing)` | Filter into a new DataBlock. `indexing` maps axis name to an expression string or a 1-D boolean mask (list/ndarray/dask). Unmentioned axes are untouched. Only a `SignalAxis` may become empty. |
| `value_get(indexing)` | Same selection, returns `(data, {axis_name: points})` instead of a DataBlock. |
| `crop_axis(name, min, max)` | Inclusive range crop (result axis is a `SignalAxis`); not for categorical. |
| `reduce_axis(name_or_names, reducer=np.max, out_quantity, out_unit)` | Collapse axes with a 1-D reducer via `da.apply_along_axis`. Multiple axes are flattened together first. Reducing all axes gives a 0-d block. |
| `rebin_reduce_axis(name, new_binsize=None, bin_edges=None, reducer=da.mean)` | Rebin by spacing or explicit edges, reduce each bin. |
| `pool(windows, reducer=np.mean, remainder="trim"\|"error"\|"pad")` | Block-pool by sample counts, e.g. `{"x": 3, "y": 3}`. `pad` fills with NaN (use `np.nanmean`). |
| `reorder_axis(name, target_index)` / `reorder_axis_inplace` | `np.moveaxis` semantics, negatives allowed. |
| `iter_axis(name, min_idx=None, max_idx=None)` | Yield slices along an axis. |
| `split_on_axis(name)` | One block per axis point; the split axis becomes a 1-point `CategoricalAxis` (so it is squeezed). Not for `UnorderedSignalAxis`. |
| `split_with_datablock(other, drop_axes=None)` | Use `other`'s integer/label values as a mask (e.g. cluster labels). Without `drop_axes`: NaN-masked, same shape. With `drop_axes` (must include every axis of `other`): packed into a new `repetition` axis (unit `#`). |
| `cluster(num_clusters=3)` | KMeans labels via dask-ml (optional dep, else ImportError). |
| `rename_quantity(new)` | New block, deep copy. |
| `save(path)` / `DataBlock.load(path)` | Zarr store (`.zarr` suffix forced). Axes stored as JSON in `attrs["metadata"]`. `load` is always lazy. |
| `to_ensemble()` | Long-form `Ensemble` via meshgrid; axes converted to `SignalAxis`. |
| arithmetic | `+ - * / ** -x`, plus reflected forms. Operand: scalar, broadcastable array, or a DataBlock with **identical axes** (name, size, points). |
| `c_op` | `NotImplementedError` |

## Ensemble methods (`NDLib/BaseStructures.py:1034`)

Constructor: `Ensemble(data: dd.DataFrame | pd.DataFrame, axes, quantity="q", unit="-")`.
The frame must have one column per axis name plus a column named `quantity`.

| Method | Behaviour |
|---|---|
| `get` / `value_get` | Same expression/mask grammar as DataBlock; implemented as `DataFrame.query` / `isin`. `get` also prunes axis points that no longer occur. |
| `reduce_axis(name(s), reducer, ...)` | Groupby over the remaining axes, aggregate with `"max"`/`"sum"`/`dd.Aggregation`/callable. Cannot reduce all axes. `out_quantity` must not clash with an axis name. |
| `rebin_reduce_axis(name, new_binsize, bin_edges, reducer=np.mean)` | Bin one axis, reduce within each bin (dask). |
| `pool(bin_sizes, reducer)` | Calls `rebin_reduce_axis` per axis. Sizes are bin **widths in axis units** (DataBlock.pool uses sample counts). |
| `rebin_axis(...)` | Rebin without reducing (works for pandas and dask). |
| `rebin_to_new_axis(...)` | Rebin and add a `repetition` axis counting duplicates per cell. |
| `to_datablock(rebin_scheme=None)` | Needs all axes `SignalAxis` with regular spacing; scatters through `sparse.COO`, fills gaps with NaN. Computes eagerly. |
| `from_datablock(db)` (classmethod) | Inverse. |
| `split_on_axis(name)` | One Ensemble per value of the axis (name is required). |
| `split_with_ensemble(other, drop_axes)` | Left-merge on common axes, split on `other`'s unique quantity values. |
| `combine(other)` | Left-merge on common axes; appends `other.quantity` as an `UnorderedSignalAxis`. pandas only (casts to `pd.DataFrame`). |
| `c_op(other, operation)` | Row-wise merge then `operation(Series, Series)`. dask only. |
| `save` / `load` | Zarr group holding `dataframe.parquet` plus JSON metadata. `load` returns a dask frame. |
| `+ - * **` | **Scalar only**, applied to the quantity column. No `/` operator. |
| `reorder_axis`, `crop_axis`, `cluster` | `NotImplementedError` (not a DataBlock-parity API yet). |

Not all Ensemble methods support both backends: check the `isinstance(..., dd.DataFrame)`
branches before assuming pandas or dask.

## Indexing expression grammar (`NDLib/IndexingUtils.py`)

- Numeric axes: `x>2`, `x<=5`, `x==3`, `x!=3`; compound `-3<=x<3`, `3>x>1`;
  scientific (`1e-3`) and `.5` literals; axis name optional in simple form (`">2"`).
- Categorical axes: `category==dog`, `label!=positive`, quoted for spaces
  (`"city=='new york'"`), name optional (`"==dog"`).
- The variable in the expression must match the axis name or `ValueError` is raised.
- Parsers: `parse_inequality_to_mask`/`_to_query`, `parse_categorical_to_mask`/`_to_query`
  (mask for DataBlock, pandas query string for Ensemble).

## Common use cases (all examples were run against the library)

Runnable, self-contained versions of these live in [`examples/`](examples/README.md)
(`uv run python examples/02_reductions.py`).

Setup shared by the examples below:

```python
import numpy as np
from NDLib import DataBlock, Ensemble, SignalAxis
from NDLib.AxesStructures import CategoricalAxis

x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
e = SignalAxis(np.linspace(0, 100, 20), "E", 2, "eV")
db = DataBlock(np.random.rand(10, 5, 20), [x, y, e], quantity="intensity", unit="counts")
```

### Statistics along axes (`reduce_axis`)

```python
# mean along E, one value per (x, y): shape (10, 5)
mean_map = db.reduce_axis("E", reducer=np.mean, out_quantity="mean_intensity", out_unit="counts")

# default reducer is np.max; any function of a 1-D array works (np.median, np.argmax, a fit...)
peak_map = db.reduce_axis("E")
peak_pos = db.reduce_axis("E", reducer=np.argmax, out_quantity="peak_index")

# several axes at once are flattened together first, so a median is the true joint median
spectrum = db.reduce_axis(["x", "y"], reducer=np.median)      # shape (20,)

# reduce everything: 0-d block
total = db.reduce_axis(["x", "y", "E"], reducer=np.sum).compute()   # numpy scalar array
```

`reduce_axis` removes the axis. To keep an axis but make it coarser, use pooling or rebinning.

### Select a subset (`get`, `value_get`)

```python
sub = db.get({"x": "x>2", "E": "10<=E<50"})      # expression strings, per axis
sub = db.get({"y": [True, False, True, False, True]})   # boolean mask, one entry per point
sub = db.get({"x": ">2"})                        # axis name may be omitted
vals, axes = db.value_get({"x": "x>5"})          # raw dask array + {"x": points, ...}

# categorical axis: select one category (the axis is then squeezed away and the
# block's quantity becomes the axis name)
param = CategoricalAxis(["center", "amplitude", "fwhm"], "param", 2, "-")
fit = DataBlock(np.random.rand(10, 5, 3), [x, y, param], quantity="fit")
center = fit.get({"param": "param==center"})     # shape (10, 5), center.quantity == "param"
```

`get` keeps the axis type. `crop_axis("E", 20, 60)` is the inclusive-range shortcut for numeric axes.

### Downsample / smooth (`pool`, `rebin_reduce_axis`)

```python
coarse = db.pool({"x": 2, "y": 5})                         # mean over 2x5 sample blocks
coarse = db.pool({"x": 3}, remainder="pad", reducer=np.nanmean)   # keep the ragged last window
binned = db.rebin_reduce_axis("E", new_binsize=10.0)       # bins ~10 eV wide, mean per bin
binned = db.rebin_reduce_axis("E", bin_edges=[0, 25, 50, 100], reducer=np.sum)
```

`pool` counts samples and keeps the remaining axes; `rebin_reduce_axis` works in axis units.

### Arithmetic and background subtraction

```python
scaled = db * 2 + 1                                       # scalars
normed = db / db.reduce_axis("E", reducer=np.max).compute()[..., None]   # broadcast array
diff = db - db                                            # DataBlock with identical axes
bg = db.reduce_axis("x", reducer=np.min).compute()        # shape (5, 20)
no_bg = db - bg[None, :, :]                               # add the missing axis by hand
```

Operands must be scalars, arrays that numpy-broadcast against `db.data`, or a DataBlock with
identical axes (name, size and points). Quantity and unit of the left operand are kept.

### Reshape, iterate, rename

```python
db.reorder_axis("E", 0)               # new block, shape (20, 10, 5)
db.reorder_axis_inplace("E", -1)      # modifies db
for spectrum in db.iter_axis("x"):    # yields arrays of shape (5, 20)
    ...
renamed = db.rename_quantity("I")
db.axis_p("E")                        # points of an axis
db.axis_obj("E").unit                 # the axis object
db.has_axis("E")
```

### Split into groups (`split_on_axis`, `split_with_datablock`)

```python
per_x = db.split_on_axis("x")         # 10 blocks, each with a 1-point CategoricalAxis "x" (squeezed)

# group by a label map: label each (x, y) pixel by where its spectrum peaks, then split
labels = db.reduce_axis("E", reducer=np.argmax, out_quantity="label")   # axes x, y
groups = db.split_with_datablock(labels, drop_axes=["x", "y"])
#   one block per unique label; the values of that group sit on a new "repetition" axis
groups = db.split_with_datablock(labels)   # without drop_axes: full shape, other values are NaN
```

`other` in `split_with_datablock` must have a subset of this block's axes. With `drop_axes`,
every axis of `other` must be listed.

### Save and load

```python
db.save("run1")                       # run1.zarr/, ".zarr" added if missing; overwrites
db = DataBlock.load("run1")           # lazy (dask); axes and quantity restored from JSON
```

### Scattered data: Ensemble

```python
import pandas as pd, dask.dataframe as dd

df = pd.DataFrame({"t": [0., 1, 2, 3], "v": [1., 2, 3, 4]})   # one column per axis + quantity
t = SignalAxis(df["t"].values, "t", 0, "s")
ens = Ensemble(dd.from_pandas(df, npartitions=1), [t], quantity="v", unit="V")

late = ens.get({"t": ">1"}).compute()          # DataFrame with rows t=2, 3

# DataBlock <-> Ensemble
ens = db.to_ensemble()                         # 1000 rows: x, y, E, intensity
means = ens.reduce_axis("E", "mean", out_quantity="m")           # groupby aggregation
pooled = ens.pool({"E": 25.0})                 # bin widths in axis units
doubled = ens * 2                              # scalar arithmetic only
grid = ens.to_datablock()                      # needs regularly spaced SignalAxes
ens.save("scatter"); Ensemble.load("scatter") # parquet inside a .zarr dir, loads lazily
```

### Evaluating lazily

```python
lazy = db.reduce_axis("E", reducer=np.mean)    # nothing computed yet (dask)
arr = lazy.compute()                           # np.ndarray; also swaps lazy.data for numpy
frame = ens.compute()                          # pd.DataFrame; also swaps ens.data for pandas
```

Gotcha: `Ensemble.compute()` replaces `ens.data` with a pandas DataFrame, and several dask-only
methods (`rebin_reduce_axis`, `pool`, `rebin_to_new_axis`, `c_op`, `split_with_ensemble`) then
fail with `'Series' object has no attribute 'map_partitions'`. Chain those first, call
`compute()` last, or keep a separate uncomputed `Ensemble`.

### Quick chooser

| I want to... | Use |
|---|---|
| mean / max / median / any statistic along an axis | `DataBlock.reduce_axis(axis, reducer=...)` |
| filter by value or label | `get({axis: "expr"})` |
| get plain arrays instead of a block | `value_get(...)` or `.compute()` |
| fewer samples, same axes | `pool` (sample counts) or `rebin_reduce_axis` (axis units) |
| trim an axis range | `crop_axis(axis, min, max)` |
| transpose | `reorder_axis` |
| one block per value / group | `split_on_axis`, `split_with_datablock` |
| persist | `save` / `load` |
| irregular samples | `Ensemble`, convert with `to_ensemble()` / `to_datablock()` |

## Repository map

```
NDLib/
  __init__.py          public exports (4 names)
  Types.py             Number, Axis1D aliases
  AxesStructures.py    Axis, SignalAxis, UnorderedSignalAxis, CategoricalAxis (264 lines)
  IndexingUtils.py     expression parsers (264 lines)
  BaseStructures.py    DataBlock, Ensemble, binning, zarr (de)serialisation (1849 lines)
examples/              7 runnable scripts, one topic each
tests/                 13 files; names say the feature (test_datablock_pool.py, test_serialization.py, ...)
pyproject.toml         deps, ruff/basedpyright/pytest config, poe tasks
.github/workflows/ci.yml   tests, ruff, basedpyright
```

Call-graph hubs (most depended on): `compute`, `SignalAxis`, `has_axis`/`axis_obj`,
`DataBlock`, `Ensemble`. `AxesStructures.py` is imported by 14 files, `BaseStructures.py` by 11.
No dependency cycles. Almost every public method starts with `has_axis` validation, then
builds a new object from `deepcopy(self.axes)`.

## Development workflow

```bash
uv sync --extra dev
uv run pytest                          # -W error, coverage via pytest-cov
uv run ruff format . && uv run ruff check .
uv run basedpyright --level error      # pinned 1.40.2; CI blocks on it
uv run poe check                       # ruff + basedpyright + ty
uv run poe quality                     # adds radon / skylos metrics
```

Style constraints enforced by ruff (preview rules on): max McCabe complexity 10, max 3
nested blocks, no `print` (use `logger`), full annotations, no boolean positional args
(`FBT`; make flags keyword-only), Google-style docstrings, double quotes, line length 88.
Test files are exempt from `ANN`, `FBT`, `assert`. CamelCase module names are allowed.
The pre-push hook and CI run tests, lint and types.

## Known rough edges

- `UnorderedSignalAxis` docstring is a `[TODO]`; its `__str__` prints `min` twice.
- `Axis.is_nav`/`ordered` are documented as unused.
- `Ensemble.to_datablock` has a `# TODO: does this work without?` around an eager compute.
- Ensemble arithmetic and several Ensemble methods are narrower than their DataBlock twins.
- README is only a stub; this file is the best overview.
