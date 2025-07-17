"""Array-checking utilities."""

import warnings

from array_api_compat import is_cupy_array, is_numpy_array

from mach._array_api import Array, array_namespace


def is_contiguous(array: Array) -> bool:
    """Check if an array is contiguous.

    Returns:
        True if the array is contiguous.
        Optimistically ASSUMES that the array is contiguous if it is not a NumPy or CuPy array,
            as many libraries do not support non-contiguous arrays.
    """
    if is_cupy_array(array) or is_numpy_array(array):
        return array.flags["C_CONTIGUOUS"]
    return True


def ensure_contiguous(array: Array, *, warn: bool = True) -> Array:
    """Ensure an array is contiguous.

    Returns:
        True if the array is contiguous.
        Optimistically ASSUMES that the array is contiguous if it is not a NumPy or CuPy array,
            as many libraries do not support non-contiguous arrays.
    """
    if is_contiguous(array):
        return array
    if warn:
        warnings.warn(
            "array is not contiguous, rearranging will add latency",
            stacklevel=2,
        )
    xp = array_namespace(array)
    return xp.ascontiguousarray(array)
