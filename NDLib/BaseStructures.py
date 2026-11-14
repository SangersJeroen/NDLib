"""Compatibility module: the structures now live in private modules.

``DataBlock`` is in ``NDLib._datablock``, ``Ensemble`` in ``NDLib._ensemble`` and the shared helpers
in ``NDLib._helpers``. This module keeps the old import paths working.
"""

from ._datablock import DASKML, DataBlock, Operand
from ._ensemble import Ensemble
from ._helpers import (
    AxisLike,
    LabelKey,
    TwoSeriesOperation,
    compute_binning,
    return_axis_idx,
)

__all__ = [
    "DASKML",
    "AxisLike",
    "DataBlock",
    "Ensemble",
    "LabelKey",
    "Operand",
    "TwoSeriesOperation",
    "compute_binning",
    "return_axis_idx",
]
