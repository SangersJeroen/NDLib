# pyright: basic
"""Statistics along one or more named axes with `reduce_axis`.

Run with: uv run python examples/02_reductions.py
"""

import numpy as np

from NDLib import DataBlock, SignalAxis


def make_block() -> DataBlock:
    rng = np.random.default_rng(0)
    x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
    y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
    energy = SignalAxis(np.linspace(0, 100, 20), "E", 2, "eV")
    return DataBlock(
        rng.random((10, 5, 20)), [x, y, energy], quantity="intensity", unit="counts"
    )


def main() -> None:
    db = make_block()

    # mean along E: one value per (x, y)
    mean_map = db.reduce_axis(
        "E", reducer=np.mean, out_quantity="mean_intensity", out_unit="counts"
    )
    print("mean map:", mean_map.shape, mean_map.quantity)

    # default reducer is np.max
    print("max map:", db.reduce_axis("E").shape)

    # where does each spectrum peak? (index along E)
    peak_index = db.reduce_axis("E", reducer=np.argmax, out_quantity="peak_index")
    print("peak index map:", peak_index.shape)

    # several axes are flattened together before the reducer runs,
    # so this is the joint median of every (x, y) at each energy
    spectrum = db.reduce_axis(["x", "y"], reducer=np.median)
    print("median spectrum:", spectrum.shape)

    # reduce everything to a single number
    total = db.reduce_axis(["x", "y", "E"], reducer=np.sum)
    print("total:", float(total.compute()))


if __name__ == "__main__":
    main()
