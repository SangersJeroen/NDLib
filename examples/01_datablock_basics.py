# pyright: basic
"""Build a DataBlock, inspect it, and do arithmetic on it.

Run with: uv run python examples/01_datablock_basics.py
"""

import numpy as np

from NDLib import DataBlock, SignalAxis


def make_block() -> DataBlock:
    """A (x, y, E) intensity cube with named axes."""
    rng = np.random.default_rng(0)
    x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
    y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
    energy = SignalAxis(np.linspace(0, 100, 20), "E", 2, "eV")
    return DataBlock(
        rng.random((10, 5, 20)), [x, y, energy], quantity="intensity", unit="counts"
    )


def main() -> None:
    db = make_block()
    print("shape:", db.shape, "dims:", db.dims, "quantity:", db.quantity)
    print("axes:", [axis.name for axis in db.axes])
    print("E axis:", db.has_axis("E"), db.axis_obj("E").unit, db.axis_p("E")[:3])

    # arithmetic keeps axes, quantity and unit, and stays lazy for dask data
    scaled = db * 2 + 1
    print("scaled mean:", float(scaled.compute().mean()))

    # a DataBlock with identical axes can be combined elementwise
    print("db - db is zero:", bool(np.all((db - db).compute() == 0)))

    # move an axis to another position (returns a new block)
    print("E first:", db.reorder_axis("E", 0).shape)

    # iterate over slices along an axis
    first = next(iter(db.iter_axis("x")))
    print("slice at first x:", first.shape)


if __name__ == "__main__":
    main()
