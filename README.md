# NDLib

Processing of n-dimensional datasets with **named, labelled axes**: select, reduce,
rebin, split and store your data by axis name instead of by array position. Built on
numpy, pandas and dask, so everything stays lazy until you ask for the result.

> v0.1.0, split from SSEELS (still private, check back soon :P)

## Install

NDLib is not on PyPI yet. Install it from GitHub (Python >= 3.11):

```bash
pip install "git+https://github.com/SangersJeroen/NDLib.git"
```

or clone it for development:

```bash
git clone https://github.com/SangersJeroen/NDLib.git
cd NDLib
uv sync --extra dev
```

`DataBlock.cluster` needs the optional `dask-ml` package (included in the `dev` extra).

## The idea

| Class | Holds | Backed by |
|---|---|---|
| `DataBlock` | a dense, regular n-d grid with one value per axis-point combination | `dask.array` / `numpy` |
| `Ensemble` | irregular or scattered samples: one row per sample, one column per axis plus the value | `dask.dataframe` / `pandas` |
| `SignalAxis` | strictly increasing numeric coordinates | |
| `UnorderedSignalAxis` | numeric coordinates in any order | |
| `CategoricalAxis` | string labels, e.g. `["center", "amplitude", "fwhm"]` | |

Every axis has a unique `name`, a `unit` and the position it occupies in the data
(`index_in_array`). Operations take axis names, so code reads like what you mean:

```python
import numpy as np
from NDLib import DataBlock, SignalAxis

x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
energy = SignalAxis(np.linspace(0, 100, 20), "E", 2, "eV")
db = DataBlock(
    np.random.rand(10, 5, 20), [x, y, energy], quantity="intensity", unit="counts"
)

# filter with expressions on named axes
sub = db.get({"x": "x>2", "E": "10<=E<50"})

# mean along E: one value per (x, y)
mean_map = db.reduce_axis("E", reducer=np.mean, out_quantity="mean_intensity")

# average 2x5 blocks of samples, or rebin an axis in its own units
coarse = db.pool({"x": 2, "y": 5})
binned = db.rebin_reduce_axis("E", new_binsize=10.0)

# arithmetic, lazy until computed
result = (db * 2 + 1).compute()

# persist as Zarr and load back lazily
db.save("run1")
db = DataBlock.load("run1")

# switch to scattered (long-form) data and back
ens = db.to_ensemble()
grid = ens.to_datablock()
```

## What you can do

- **Select**: `get` and `value_get` with expressions (`x>2`, `-3<=x<3`, `label==dog`) or boolean masks, `crop_axis`
- **Reduce**: `reduce_axis` with any function of a 1-D array (mean, median, argmax, a fit, ...), over one or several axes
- **Downsample**: `pool` (by sample count), `rebin_reduce_axis` (by axis units)
- **Reshape**: `reorder_axis`, `iter_axis`, `rename_quantity`
- **Split**: `split_on_axis`, `split_with_datablock` / `split_with_ensemble` (group by a label map, for example cluster labels)
- **Combine**: arithmetic between blocks with identical axes, `Ensemble.combine`, `Ensemble.c_op`
- **Store**: `save` / `load` as Zarr (Ensembles store their frame as Parquet inside)

## Documentation

- [`LLM_GUIDE.md`](LLM_GUIDE.md): compact API reference with every method, the invariants to know about,
  and a recipe per common use case. Written to be read by people and by LLMs.
- [`examples/`](examples/README.md): seven short, self-contained scripts you can run
  (`uv run python examples/02_reductions.py`).
- Docstrings in `NDLib/BaseStructures.py` and `NDLib/AxesStructures.py`.

## Things to know

- Almost every method returns a **new** object; the exceptions are `compute()` and `reorder_axis_inplace`.
- `compute()` replaces the object's lazy data by numpy / pandas in place. Prefer computing at the end of a pipeline.
- Axes of size 1 are squeezed away when a `DataBlock` is built.
- `Ensemble` is the less complete sibling: arithmetic is scalar-only and `reorder_axis`, `crop_axis` and `cluster` are not implemented yet.

## Development

```bash
uv run pytest                      # tests, warnings are errors, 80% coverage gate
uv run ruff format . && uv run ruff check .
uv run basedpyright --level error  # strict type check, version pinned
uv run poe check                   # ruff + basedpyright + ty
```

CI runs the tests, `ruff` and `basedpyright` on pushes to `main` and on pull requests to `main`.
