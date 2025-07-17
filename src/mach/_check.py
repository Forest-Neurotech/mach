"""Array-checking utilities."""

import warnings
from typing import cast

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
        assert hasattr(array, "flags"), "numpy or cupy array should have flags"
        # Type ignore because numpy/cupy flags objects support dict-like access but aren't Mapping types
        assert "C_CONTIGUOUS" in array.flags, "numpy or cupy array should have C_CONTIGUOUS flag"  # type: ignore[operator]
        return cast(bool, array.flags["C_CONTIGUOUS"])  # type: ignore[index]
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
    return xp.ascontiguousarray(array)  # type: ignore[attr-defined]  # Assume numpy/cupy
