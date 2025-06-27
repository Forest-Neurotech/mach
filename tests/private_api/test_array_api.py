"""Test the array API utilities."""

from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace

from mach._array_api import ArrayAPIConformant, DLPackDevice


def test_array_protocol(xp):
    """Test that the ArrayAPIConformant protocol accurately describes supported array libraries."""
    arr = xp.array([[1, 2], [3, 4]])

    # Type checking - verify arr matches protocol
    assert isinstance(arr, ArrayAPIConformant)

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
