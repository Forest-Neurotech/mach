"""Tests for beamforming with Array API-compatible backends."""

from typing import Any

import numpy as np
import pytest
from array_api_compat import array_namespace, is_cupy_namespace, is_jax_namespace, is_numpy_namespace

import mach
from mach.kernel import beamform


def create_test_data(xp: Any) -> tuple[Any, Any, Any, Any, Any, int, int, int, int]:
    """Create test data using the specified array library."""
    # Sample dimensions
    n_receive_elements = 4
    n_samples = 100
    n_frames = 2
    n_scan_points = 8

    # Create RF data (shape: n_receive_elements, n_samples, n_frames)
    if is_numpy_namespace(xp) or is_cupy_namespace(xp):
        rf = xp.random.normal(size=(n_receive_elements, n_samples, n_frames)).astype(xp.float32)
        rf_complex = (
            rf + 1j * xp.random.normal(size=(n_receive_elements, n_samples, n_frames)).astype(xp.float32, copy=False)
        ).astype(xp.complex64, copy=False)
        coords = xp.random.normal(size=(n_receive_elements, 3)).astype(xp.float32)
        grid = xp.random.normal(size=(n_scan_points, 3)).astype(xp.float32)
        idt_matrix = xp.zeros(n_scan_points).astype(xp.float32)
    elif is_jax_namespace(xp):
        import jax

        rf = jax.random.normal(key=jax.random.PRNGKey(0), shape=(n_receive_elements, n_samples, n_frames)).astype(
            xp.float32
        )
        rf_complex = (
            rf
            + 1j
            * jax.random.normal(key=jax.random.PRNGKey(1), shape=(n_receive_elements, n_samples, n_frames)).astype(
                xp.float32, copy=False
            )
        ).astype(xp.complex64, copy=False)
        coords = jax.random.normal(key=jax.random.PRNGKey(2), shape=(n_receive_elements, 3)).astype(xp.float32)
        grid = jax.random.normal(key=jax.random.PRNGKey(3), shape=(n_scan_points, 3)).astype(xp.float32)
        idt_matrix = xp.zeros(n_scan_points).astype(xp.float32)
    else:
        raise ValueError(f"Unsupported array library: {xp}")

    return rf, rf_complex, coords, grid, idt_matrix, n_frames, n_receive_elements, n_samples, n_scan_points


@pytest.fixture
def test_data(xp):
    """Fixture providing test data for the given array backend."""
    return create_test_data(xp)


def test_beamform_with_real_data(xp, test_data):
    """Test beamforming with real data."""
    rf, _, coords, grid, idt_matrix, nframes, _, _, n_scan_points = test_data

    out = None
    if is_jax_namespace(xp):
        # JAX does not support in-place operations, so we need to create a writable output array
        import cupy as cp

        out = cp.zeros((n_scan_points, nframes), dtype=rf.dtype)

    # Test with real data
    output_real = beamform(
        channel_data=rf,
        rx_coords_m=coords,
        scan_coords_m=grid,
        tx_wave_arrivals_s=idt_matrix,
        out=out,
        f_number=2.0,
        rx_start_s=0.0,
        sampling_freq_hz=1e6,
        sound_speed_m_s=1500.0,
    )

    # Verify output shape and type
    assert output_real.shape == (n_scan_points, nframes)

    # If beamform created output_real, it should have the same type/namespace as input
    if out is None:
        assert array_namespace(output_real) == array_namespace(rf)


def test_beamform_with_complex_data(xp, test_data):
    """Test beamforming with complex data."""
    _, rf_complex, coords, grid, idt_matrix, nframes, _, _, n_scan_points = test_data

    out = None
    if is_jax_namespace(xp):
        # JAX does not support in-place operations, so we need to create a writable output array
        import cupy as cp

        out = cp.zeros((n_scan_points, nframes), dtype=rf_complex.dtype)

    # Test with complex data
    output_complex = beamform(
        channel_data=rf_complex,
        rx_coords_m=coords,
        scan_coords_m=grid,
        tx_wave_arrivals_s=idt_matrix,
        out=out,
        f_number=2.0,
        rx_start_s=0.0,
        sampling_freq_hz=1e6,
        sound_speed_m_s=1500.0,
        modulation_freq_hz=5e6,
    )

    # Verify output shape and type
    assert output_complex.shape == (n_scan_points, nframes)

    # If beamform created output_complex, it should have the same type/namespace as input
    if out is None:
        assert array_namespace(output_complex) == array_namespace(rf_complex)


@pytest.mark.parametrize(
    "tukey_alpha",
    [0.0, 0.5, 1.0],
)
def test_beamform_apodization(xp, test_data, tukey_alpha):
    """Test beamforming with and without apodization."""
    _, rf_complex, coords, grid, idt_matrix, nframes, _, _, n_scan_points = test_data

    out = None
    if is_jax_namespace(xp):
        # JAX does not support in-place operations, so we need to create a writable output array
        import cupy as cp

        out = cp.zeros((n_scan_points, nframes), dtype=rf_complex.dtype)

    # Test with and without apodization
    output = beamform(
        channel_data=rf_complex,
        rx_coords_m=coords,
        scan_coords_m=grid,
        tx_wave_arrivals_s=idt_matrix,
        out=out,
        f_number=2.0,
        rx_start_s=0.0,
        sampling_freq_hz=1e6,
        sound_speed_m_s=1500.0,
        modulation_freq_hz=5e6,
        tukey_alpha=tukey_alpha,
    )

    # Verify output shape
    assert output.shape == (n_scan_points, nframes)

    # If beamform created output, it should have the same type/namespace as input
    if out is None:
        assert array_namespace(output) == array_namespace(rf_complex)


def test_beamform_mixed_cpu_gpu_arrays(xp, test_data):
    """Test beamforming with inputs on both CPU."""
    if is_numpy_namespace(xp):
        pytest.skip("Numpy is only on CPU, so this test is not relevant")

    _, rf_complex, coords, grid, idt_matrix, nframes, _, _, n_scan_points = test_data

    out = None
    if is_jax_namespace(xp):
        # JAX does not support in-place operations, so we need to create a writable output array
        import cupy as cp

        out = cp.zeros((n_scan_points, nframes), dtype=rf_complex.dtype)

    # Data should already be on the GPU
    # Try moving one of the inputs to the CPU to check that it works
    # just with a warning
    if is_cupy_namespace(xp):
        import cupy as cp

        grid = cp.asnumpy(grid)
    else:
        grid = np.array(grid)

    with pytest.warns(UserWarning, match=".*latency.*"):
        output = beamform(
            channel_data=rf_complex,
            rx_coords_m=coords,
            scan_coords_m=grid,
            tx_wave_arrivals_s=idt_matrix,
            out=out,
            f_number=2.0,
            rx_start_s=0.0,
            sampling_freq_hz=1e6,
            sound_speed_m_s=1500.0,
            modulation_freq_hz=5e6,
        )

    # Verify output shape
    assert output.shape == (n_scan_points, nframes)

    # If beamform created output, it should have the same type/namespace as input
    if out is None:
        assert array_namespace(output) == array_namespace(rf_complex)


@pytest.mark.parametrize(
    "f_number,alpha",
    [
        (1.0, 0.2),
        (2.0, 0.5),
        (3.0, 0.8),
    ],
)
def test_beamform_parameters(xp, test_data, f_number, alpha):
    """Test beamforming with different parameter values."""
    _, rf_complex, coords, grid, idt_matrix, nframes, _, _, n_scan_points = test_data

    out = None
    if is_jax_namespace(xp):
        # JAX does not support in-place operations, so we need to create a writable output array
        import cupy as cp

        out = cp.zeros((n_scan_points, nframes), dtype=rf_complex.dtype)

    # Test with different parameters
    output = beamform(
        channel_data=rf_complex,
        rx_coords_m=coords,
        scan_coords_m=grid,
        tx_wave_arrivals_s=idt_matrix,
        out=out,
        f_number=f_number,
        rx_start_s=0.0,
        sampling_freq_hz=1e6,
        sound_speed_m_s=1500.0,
        modulation_freq_hz=5e6,
        tukey_alpha=alpha,
    )

    # Verify output shape
    assert output.shape == (n_scan_points, nframes)

    # If beamform created output, it should have the same type/namespace as input
    if out is None:
        assert array_namespace(output) == array_namespace(rf_complex)


def test_data_shape_errors(xp, test_data):
    """Test that the beamform function raises an error if the data shapes are not correct."""
    _, rf_complex, coords, grid, idt_matrix, nframes, _, _, n_scan_points = test_data

    out = None
    if is_jax_namespace(xp):
        # JAX does not support in-place operations, so we need to create a writable output array
        import cupy as cp

        out = cp.zeros((n_scan_points, nframes), dtype=rf_complex.dtype)

    # Test with different parameters
    num_rx_elements = rf_complex.shape[0]
    with pytest.raises(RuntimeError, match="Dimension mismatch in receive elements"):
        _ = beamform(
            channel_data=rf_complex[: num_rx_elements // 2],
            rx_coords_m=coords,
            scan_coords_m=grid,
            tx_wave_arrivals_s=idt_matrix,
            out=out,
            f_number=2.0,
            rx_start_s=0.0,
            sampling_freq_hz=1e6,
            sound_speed_m_s=1500.0,
            modulation_freq_hz=5e6,
        )

    # Test with boolean data
    with pytest.raises(TypeError, match="channel_data must be array with dtype=numeric"):
        _ = beamform(
            channel_data=rf_complex.astype(bool),
            rx_coords_m=coords,
            scan_coords_m=grid,
            tx_wave_arrivals_s=idt_matrix,
            out=out,
            f_number=2.0,
            rx_start_s=0.0,
            sampling_freq_hz=1e6,
            sound_speed_m_s=1500.0,
            modulation_freq_hz=5e6,
        )


def test_nvcc_version():
    """Test that the NVCC version is correctly exposed."""
    assert hasattr(mach.kernel, "__nvcc_version__")
    assert mach.kernel.__nvcc_version__ is not None
    # version-style: X.Y.Z
    major_version_str, minor_version_str, _ = mach.kernel.__nvcc_version__.split(".")
    major_version = int(major_version_str)
    minor_version = int(minor_version_str)
    assert major_version >= 12, f"NVCC version {major_version} is less than required minimum 11"
    assert minor_version >= 0, f"NVCC version {minor_version} is less than required minimum 0"
