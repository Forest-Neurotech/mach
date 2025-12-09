"""Array-API utilities."""

from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

from array_api_compat import array_namespace as xpc_array_namespace


class DLPackDevice(int, Enum):
    """Enum for the different DLPack device types.

    Port of:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L76-L80
    """

    CPU = 1
    CUDA = 2


@runtime_checkable
class LinAlg(Protocol):
    """Protocol for linear algebra extension conforming to Array API standard.

    This is an optional extension - not all array libraries implement it.
    See: https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html
    """

    def vector_norm(self, x: "Array", *, axis: Any = None, keepdims: bool = False, ord: Any = None) -> "Array": ...  # noqa: A002


@runtime_checkable
class _ArrayNamespace(Protocol):
    """Protocol for array namespaces that conform to the Array API standard.

    This covers the common operations and data types used throughout the mach codebase.
    Based on the Array API specification: https://data-apis.org/array-api/latest/

    Note: This base protocol does NOT include the optional linalg extension.
    Use _ArrayNamespaceWithLinAlg for namespaces that have the extension.
    """

    # Data types
    float32: Any
    float64: Any
    complex64: Any
    complex128: Any
    int8: Any
    int16: Any
    int32: Any
    int64: Any
    uint8: Any
    uint16: Any
    uint32: Any
    uint64: Any

    # Core linear algebra functions (part of main API)
    def matmul(self, x1: "Array", x2: "Array") -> "Array": ...
    def tensordot(self, x1: "Array", x2: "Array", *, axes: Any = 2) -> "Array": ...
    def vecdot(self, x1: "Array", x2: "Array", *, axis: int = -1) -> "Array": ...

    # Mathematical functions
    def abs(self, x: "Array") -> "Array": ...
    def cos(self, x: "Array") -> "Array": ...
    def sign(self, x: "Array") -> "Array": ...
    def sin(self, x: "Array") -> "Array": ...
    def sqrt(self, x: "Array") -> "Array": ...
    def sum(self, x: "Array", *, axis: Any = None, keepdims: bool = False) -> "Array": ...

    # Array creation and manipulation
    def asarray(self, obj: Any, *, dtype: Any = None, device: Any = None, copy: bool = False) -> "Array": ...
    def stack(self, arrays: Any, *, axis: int = 0) -> "Array": ...
    def zeros(self, shape: Any, *, dtype: Any = None, device: Any = None) -> "Array": ...


@runtime_checkable
class _ArrayNamespaceWithLinAlg(_ArrayNamespace, Protocol):
    """Extended _ArrayNamespace protocol that includes the linear algebra extension.

    Use this when you know the array namespace supports the linalg extension.
    Most code should use the base _ArrayNamespace and check with hasattr(xp, 'linalg').
    """

    linalg: LinAlg


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
    def __mul__(self, other: Any) -> "Array": ...
    def __truediv__(self, other: Any) -> "Array": ...
    def __neg__(self) -> "Array": ...
    def __pos__(self) -> "Array": ...
    def __lt__(self, other: Any) -> "Array": ...
    def __gt__(self, other: Any) -> "Array": ...
    def __getitem__(self, key: Any) -> "Array": ...


def array_namespace(*arrays: Any) -> _ArrayNamespace | _ArrayNamespaceWithLinAlg:
    """Typed wrapper around array_api_compat.array_namespace.

    Returns the array namespace for the given arrays with proper type hints.
    This resolves static typing issues by providing an ArrayNamespace protocol.

    Args:
        *arrays: Arrays to get the namespace for

    Returns:
        Union[_ArrayNamespace, _ArrayNamespaceWithLinAlg]: The appropriate array namespace (numpy, cupy, jax.numpy, etc.)
                                                          May or may not include the linalg extension.

    Note:
        The linalg extension may not be available in all array libraries.
        Code should check for its existence using hasattr(xp, 'linalg') before use.

    Examples:
        >>> xp = array_namespace(arr)
        >>> if hasattr(xp, 'linalg'):
        ...     norm = xp.linalg.vector_norm(arr)
        ... else:
        ...     # Fallback implementation
        ...     norm = xp.sqrt(xp.sum(arr * arr, axis=-1))
    """
    return cast(_ArrayNamespace | _ArrayNamespaceWithLinAlg, xpc_array_namespace(*arrays))
