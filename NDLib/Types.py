from typing import TypeAlias
import numpy as np
from numpy.typing import ArrayLike


Number: TypeAlias = np.float64 | np.int32 | np.float128 | int | float
Axis1D: TypeAlias = ArrayLike
