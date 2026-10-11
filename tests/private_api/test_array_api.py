"""Test the array API utilities."""

import pytest
from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace
from array_api_extra.testing import assert_equal

from mach._array_api import (
    Array,
    DLPackDevice,
    _ArrayNamespace,
    _ArrayNamespaceWithLinAlg,
    array_namespace,
    vector_norm,
)


def test_array_protocol(xp):
    """Test that the Array protocol accurately describes supported array libraries."""
    arr = xp.array([[1, 2], [3, 4]])

    # Type checking - verify arr matches protocol
    assert isinstance(arr, Array)
    namespace = array_namespace(arr)
    assert isinstance(namespace, _ArrayNamespace)
    assert isinstance(namespace, _ArrayNamespaceWithLinAlg)

    # Check device type
    device_type, _ = arr.__dlpack_device__()
    if is_numpy_namespace(xp):
        assert device_type == DLPackDevice.CPU
    elif is_jax_namespace(xp):
        import jax

        if jax.devices("gpu"):
            assert device_type == DLPackDevice.CUDA
        else:
            assert device_type == DLPackDevice.CPU
    elif is_cupy_namespace(xp):
        assert device_type == DLPackDevice.CUDA


@pytest.mark.parametrize(
    ("vectors", "expected"),
    [
        ([[3.0, 4.0], [5.0, 12.0]], [5.0, 13.0]),
        ([[3.0 + 4.0j, 0.0j], [0.0j, 5.0 + 12.0j]], [5.0, 13.0]),
    ],
    ids=["real", "complex"],
)
def test_vector_norm(xp, vectors, expected):
    """Test vector norms across supported array libraries."""
    vectors = xp.asarray(vectors)

    result = vector_norm(vectors, axis=-1)

    assert_equal(result, xp.asarray(expected))
