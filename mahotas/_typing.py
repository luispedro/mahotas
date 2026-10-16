"""Shared types for the Python API (not runtime argument validation)."""

from os import PathLike
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[Any]
BorderMode: TypeAlias = Literal[
    'nearest', 'wrap', 'reflect', 'mirror', 'constant', 'ignore',
]
StructuringElement: TypeAlias = int | Array | None
Path: TypeAlias = str | PathLike[str]
ScalarT = TypeVar('ScalarT', bound=np.generic)
