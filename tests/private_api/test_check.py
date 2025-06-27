"""Unit tests for mach._check module."""

import numpy as np
import pytest

from mach._check import ensure_contiguous, is_contiguous


def test_contiguous_numpy_array():
    """Test that contiguous numpy arrays are correctly identified."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert is_contiguous(arr)
    assert ensure_contiguous(arr) is arr

    assert not is_contiguous(arr.T)
    with pytest.warns(UserWarning, match="array is not contiguous"):
        assert ensure_contiguous(arr.T) is not arr.T


def test_fortran_order_array():
    """Test that Fortran-ordered arrays are not considered C-contiguous."""
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order="F")
    # Fortran order is not C-contiguous
    assert not is_contiguous(arr)
    with pytest.warns(UserWarning, match="array is not contiguous"):
        assert ensure_contiguous(arr) is not arr
