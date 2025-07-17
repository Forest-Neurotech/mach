"""Array-API utilities."""

from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

# Import the original array_namespace for our wrapper
from array_api_compat import array_namespace as _array_namespace


class DLPackDevice(int, Enum):
    """Enum for the different DLPack device types.

    Port of:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L76-L80
    """

    CPU = 1
    CUDA = 2


@runtime_checkable
class LinAlg(Protocol):
    """Protocol for linear algebra namespace conforming to Array API standard."""

    def vector_norm(self, x: "Array", *, axis: Any = None, keepdims: bool = False, ord: Any = None) -> "Array": ...  # noqa: A002


@runtime_checkable
class ArrayNamespace(Protocol):
    """Protocol for array namespaces that conform to the Array API standard.

    This covers the common operations and data types used throughout the mach codebase.
    Based on the Array API specification: https://data-apis.org/array-api/latest/
    """

    # Data types
    float32: Any
    complex64: Any
    complex128: Any

    # Linear algebra module
    linalg: LinAlg

    # Mathematical functions
    def abs(self, x: "Array") -> "Array": ...
    def cos(self, x: "Array") -> "Array": ...
    def sign(self, x: "Array") -> "Array": ...
    def sin(self, x: "Array") -> "Array": ...

    # Array creation and manipulation
    def asarray(self, obj: Any, *, dtype: Any = None, device: Any = None, copy: bool = False) -> "Array": ...
    def stack(self, arrays: Any, *, axis: int = 0) -> "Array": ...
    def zeros(self, shape: Any, *, dtype: Any = None, device: Any = None) -> "Array": ...

    # Array operations
    def vecdot(self, x1: "Array", x2: "Array", *, axis: int = -1) -> "Array": ...


@runtime_checkable
class Array(Protocol):
    """Protocol for arrays that conform to the Array API standard.

    This is a lightweight implementation that covers the basic operations
    needed by the mach codebase.  It will eventually be replaced by one of the
    following:
    https://github.com/magnusdk/spekk/commit/d17d5bbd3e2beac97142a9397ce25942b787a7ed
    https://github.com/data-apis/array-api/pull/589/
    https://github.com/data-apis/array-api-typing
    """

    dtype: Any
    shape: tuple[int, ...]

    def __dlpack_device__(self) -> tuple[int, int]: ...

    # Basic operations used in the codebase
    def __add__(self, other: Any) -> "Array": ...
    def __sub__(self, other: Any) -> "Array": ...
    def __truediv__(self, other: Any) -> "Array": ...
    def __getitem__(self, key: Any) -> "Array": ...
    def reshape(self, shape: Any) -> "Array": ...
    def astype(self, dtype: Any, *, copy: bool = True) -> "Array": ...

    @property
    def T(self) -> "Array": ...


def array_namespace(*arrays: Any) -> ArrayNamespace:
    """Typed wrapper around array_api_compat.array_namespace.

    Returns the array namespace for the given arrays with proper type hints.
    This resolves static typing issues by providing an ArrayNamespace protocol.

    Args:
        *arrays: Arrays to get the namespace for

    Returns:
        ArrayNamespace: The appropriate array namespace (numpy, cupy, jax.numpy, etc.)
    """
    return cast(ArrayNamespace, _array_namespace(*arrays))
