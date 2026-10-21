# pyright: basic
"""Split a DataBlock into groups.

Run with: uv run python examples/05_splitting.py
"""

import numpy as np

from NDLib import DataBlock, SignalAxis


def main() -> None:
    rng = np.random.default_rng(0)
    x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
    y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
    energy = SignalAxis(np.linspace(0, 100, 20), "E", 2, "eV")
    db = DataBlock(
        rng.random((10, 5, 20)), [x, y, energy], quantity="intensity", unit="counts"
    )

    # one block per point of an axis; the split axis becomes a 1-point
    # categorical axis and is squeezed away
    per_x = db.split_on_axis("x")
    print("split_on_axis:", len(per_x), per_x[0].shape)

    # group (x, y) pixels by where their spectrum peaks, then split on that label map
    labels = db.reduce_axis("E", reducer=np.argmax, out_quantity="label")
    groups = db.split_with_datablock(labels, drop_axes=["x", "y"])
    print("groups:", len(groups))
    # each group has the E axis plus a new "repetition" axis holding its pixels
    print("first group axes:", [axis.name for axis in groups[0].axes])

    # without drop_axes the blocks keep the full shape, other pixels are NaN
    masked = db.split_with_datablock(labels)
    print("masked:", masked[0].shape)


if __name__ == "__main__":
    main()
