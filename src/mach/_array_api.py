"""Array-API utilities."""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DLPackDevice(int, Enum):
    """Enum for the different DLPack device types.

    Port of:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L76-L80
    """

    CPU = 1
    CUDA = 2


@runtime_checkable
class ArrayAPIConformant(Protocol):
    """Protocol for arrays that conform to the Array API standard.

    This is a lightweight implementation that will eventually be replaced by
    one of the following:
    https://github.com/magnusdk/spekk/commit/d17d5bbd3e2beac97142a9397ce25942b787a7ed
    https://github.com/data-apis/array-api/pull/589/
    https://github.com/data-apis/array-api-typing
    """

    dtype: Any
    shape: tuple[int, ...]

    def __dlpack_device__(self) -> tuple[int, int]: ...
