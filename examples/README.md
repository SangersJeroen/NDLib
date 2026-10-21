# Examples

Self-contained scripts, each runnable on its own from the repository root:

```bash
uv run python examples/02_reductions.py
```

| Script | Shows |
|---|---|
| `01_datablock_basics.py` | building a `DataBlock`, inspecting axes, arithmetic, reordering |
| `02_reductions.py` | mean / max / median / argmax along axes with `reduce_axis` |
| `03_selection.py` | `get`, `value_get`, `crop_axis`, categorical selection |
| `04_downsampling.py` | `pool` and `rebin_reduce_axis` |
| `05_splitting.py` | `split_on_axis`, `split_with_datablock` |
| `06_save_and_load.py` | Zarr persistence for `DataBlock` and `Ensemble` |
| `07_ensemble.py` | scattered data and `DataBlock` <-> `Ensemble` conversion |

See [`LLM_GUIDE.md`](../LLM_GUIDE.md) for the full API overview.
