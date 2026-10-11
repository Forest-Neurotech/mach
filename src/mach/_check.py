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
        assert hasattr(array, "flags"), "numpy or cupy array should have flags"
        # Type ignore because numpy/cupy flags objects support dict-like access
        return array.flags["C_CONTIGUOUS"]  # ty: ignore[not-subscriptable]
    return True


def try_contiguous(array: Array, *, warn: bool = True) -> Array:
    """Return a contiguous array when contiguity can be determined.

    Arrays without NumPy or CuPy flags are assumed to be contiguous and
    returned unchanged.

    Args:
        array:
            Input array.
        warn:
            Whether to warn before copying a non-contiguous array.

    Returns:
        The input array or a contiguous copy.

    Raises:
        ValueError:
            If the array is known to be non-contiguous but its namespace
            cannot create a contiguous copy.
    """
    if is_contiguous(array):
        return array
    if warn:
        warnings.warn(
            "array is not contiguous, rearranging will add latency",
            stacklevel=2,
        )
    xp = array_namespace(array)
    ascontiguousarray = getattr(xp, "ascontiguousarray", None)
    if callable(ascontiguousarray):
        return ascontiguousarray(array)

    raise ValueError(f"array namespace {xp} does not support `ascontiguousarray`")
