# pyright: basic
"""Select subsets with expression strings, boolean masks and categories.

Run with: uv run python examples/03_selection.py
"""

import numpy as np

from NDLib import DataBlock, SignalAxis
from NDLib.AxesStructures import CategoricalAxis


def main() -> None:
    rng = np.random.default_rng(0)
    x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
    y = SignalAxis(np.linspace(0, 4, 5), "y", 1, "um")
    energy = SignalAxis(np.linspace(0, 100, 20), "E", 2, "eV")
    db = DataBlock(
        rng.random((10, 5, 20)), [x, y, energy], quantity="intensity", unit="counts"
    )

    # expression strings, one per axis; axes not mentioned are kept whole
    sub = db.get({"x": "x>2", "E": "10<=E<50"})
    print("expression:", sub.shape)

    # the axis name may be omitted in a simple comparison
    print("short form:", db.get({"x": ">2"}).shape)

    # boolean mask with one entry per axis point
    print("mask:", db.get({"y": [True, False, True, False, True]}).shape)

    # inclusive numeric range
    print("crop:", db.crop_axis("E", 20, 60).shape)

    # raw arrays instead of a DataBlock
    values, axes = db.value_get({"x": "x>5"})
    print("value_get:", values.shape, sorted(axes))

    # categorical axis: select one category, the axis is squeezed away and the
    # quantity becomes the axis name
    param = CategoricalAxis(["center", "amplitude", "fwhm"], "param", 2, "-")
    fit = DataBlock(rng.random((10, 5, 3)), [x, y, param], quantity="fit")
    center = fit.get({"param": "param==center"})
    print("category:", center.shape, center.quantity)


if __name__ == "__main__":
    main()
