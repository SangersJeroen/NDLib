# pyright: basic
"""Work with scattered data in an Ensemble and convert to and from a DataBlock.

Run with: uv run python examples/07_ensemble.py
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd

from NDLib import DataBlock, Ensemble, SignalAxis


def main() -> None:
    # one column per axis plus a column named like the quantity
    df = pd.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0, 4.0]})
    t = SignalAxis(df["t"].to_numpy(), "t", 0, "s")
    ens = Ensemble(dd.from_pandas(df, npartitions=1), [t], quantity="v", unit="V")
    print(ens.get({"t": ">1"}).compute())

    # DataBlock -> Ensemble: one row per grid point
    rng = np.random.default_rng(0)
    x = SignalAxis(np.linspace(0, 9, 10), "x", 0, "um")
    energy = SignalAxis(np.linspace(0, 100, 20), "E", 1, "eV")
    db = DataBlock(rng.random((10, 20)), [x, energy], quantity="intensity")
    scattered = db.to_ensemble()

    # aggregate over an axis, bin an axis, do scalar arithmetic
    print(
        "mean over E:",
        scattered.reduce_axis("E", "mean", out_quantity="m").compute().shape,
    )
    print("pooled E:", scattered.pool({"E": 25.0}).compute().shape)
    print("doubled:", (scattered * 2).compute().shape)

    # and back onto the regular grid (needs regularly spaced SignalAxes)
    print("round trip:", scattered.to_datablock().shape)

    # compute() at the end: it turns the dask frame into a pandas DataFrame
    print(type(scattered.compute()).__name__)


if __name__ == "__main__":
    main()
