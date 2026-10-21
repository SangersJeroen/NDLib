# pyright: basic
"""Make data coarser with `pool` (sample counts) and `rebin_reduce_axis` (axis units).

Run with: uv run python examples/04_downsampling.py
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

    # average 2x5 blocks of samples; other axes are untouched
    print("pool:", db.pool({"x": 2, "y": 5}).shape)

    # 10 is not divisible by 3: trim (default) drops the leftover, pad keeps it as NaN
    print("trim:", db.pool({"x": 3}).shape)
    print("pad:", db.pool({"x": 3}, remainder="pad", reducer=np.nanmean).shape)

    # rebin in axis units: bins of roughly 10 eV, mean per bin
    print("binsize:", db.rebin_reduce_axis("E", new_binsize=10.0).shape)

    # or with explicit edges and another reducer
    binned = db.rebin_reduce_axis("E", bin_edges=[0, 25, 50, 100], reducer=np.sum)
    print("edges:", binned.shape, binned.axis_p("E"))


if __name__ == "__main__":
    main()
