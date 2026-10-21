# pyright: basic
"""Persist a DataBlock and an Ensemble as Zarr stores and load them back lazily.

Run with: uv run python examples/06_save_and_load.py
"""

from pathlib import Path
import tempfile

import numpy as np

from NDLib import DataBlock, Ensemble, SignalAxis


def main() -> None:
    rng = np.random.default_rng(0)
    x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
    y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
    db = DataBlock(rng.random((10, 5)), [x, y], quantity="intensity", unit="counts")

    with tempfile.TemporaryDirectory() as tmp:
        # a ".zarr" suffix is added when missing
        db.save(Path(tmp) / "block")
        loaded = DataBlock.load(Path(tmp) / "block")
        print("loaded:", loaded.shape, loaded.quantity, type(loaded.data).__name__)
        print("same data:", bool(np.allclose(loaded.compute(), db.compute())))

        db.to_ensemble().save(Path(tmp) / "scatter")
        ens = Ensemble.load(Path(tmp) / "scatter")
        print("ensemble rows:", len(ens.compute()))


if __name__ == "__main__":
    main()
