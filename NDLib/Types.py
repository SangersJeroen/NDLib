from typing import TypeAlias

import numpy as np

Number: TypeAlias = np.float64 | np.int32 | np.float128 | int | float
Axis1D: TypeAlias = np.ndarray[tuple[int], np.dtype]
