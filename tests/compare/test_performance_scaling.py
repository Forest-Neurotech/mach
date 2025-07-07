"""Performance scaling tests for mach beamformer.

These tests measure how mach performance scales with different dataset dimensions:
- Number of voxels (via grid resolution)
- Number of receive elements
- Ensemble size (number of frames)

Usage:
    # Run scaling tests without benchmarking
    pytest tests/compare/test_performance_scaling.py --benchmark-disable

    # Benchmark scaling performance
    pytest tests/compare/test_performance_scaling.py --benchmark-histogram --benchmark-sort=mean
"""

import numpy as np
import pytest
from einops import rearrange

# Import PyMUST for data conversion
pytest.importorskip("pymust")
import pymust

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from mach import geometry, wavefront
from mach.kernel import nb_beamform


@pytest.fixture(scope="session")
def pymust_iq_data(pymust_data, pymust_params):
    """Extract RF/IQ data from PyMUST data file."""
    mat_data = pymust_data
    rf_data = mat_data["RF"].astype(float)
    iq_data = pymust.rf2iq(rf_data, pymust_params)
    return np.ascontiguousarray(iq_data, dtype=np.complex64)


@pytest.fixture(scope="session")
def base_scaling_data(pymust_iq_data, pymust_element_positions, pymust_params):
    """Base data for scaling tests - single frame, baseline resolution."""

    # Use single frame for baseline
    # single_frame_data = pymust_iq_data[:, :, :1].copy()

    # Base grid with 1e-4 resolution (roughly 100 μm spacing)
    n_x = 251  # Same as original PyMUST tests
    n_z = 251

    x_range = np.linspace(-1.25e-2, 1.25e-2, num=n_x, endpoint=True)
    y_range = np.array([0.0])
    z_range = np.linspace(1e-2, 3.5e-2, num=n_z, endpoint=True)

    x_grid, z_grid = np.meshgrid(x_range, z_range, indexing="ij")
    y_grid = np.zeros_like(x_grid)

    grid_points = np.stack([x_grid.flat, y_grid.flat, z_grid.flat], axis=-1)

    # Compute transmit arrivals for plane wave
    sound_speed_m_s = pymust_params["c"]
    direction = np.asarray(geometry.ultrasound_angles_to_cartesian(0, 0))

    transmit_arrivals_s = (
        wavefront.plane(
            origin_m=np.array([0, 0, 0]),
            points_m=grid_points,
            direction=direction,
        )
        / sound_speed_m_s
    )

    # Reorder IQ data for mach format
    iq_data_reordered = np.ascontiguousarray(
        rearrange(pymust_iq_data, "n_samples n_elements n_frames -> n_elements n_samples n_frames"),
        dtype=np.complex64,
    )

    return {
        "iq_data": iq_data_reordered,
        "element_positions": pymust_element_positions,
        "scan_coords_m": grid_points.astype(np.float32),
        "transmit_arrivals_s": transmit_arrivals_s.flatten().astype(np.float32),
        "params": pymust_params,
        "grid_shape": (n_x, len(y_range), n_z),
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
    }


@pytest.mark.benchmark(
    group="scaling_voxels",
    min_time=0.1,
    max_time=1.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.parametrize(
    "grid_resolution",
    [
        pytest.param(1e-4, id="res_1e-4"),
        pytest.param(5e-5, id="res_5e-5"),
        pytest.param(2e-5, id="res_2e-5"),
        pytest.param(1e-5, id="res_1e-5"),
    ],
)
def test_scaling_voxels(benchmark, base_scaling_data, grid_resolution):
    """Test performance scaling with number of voxels (grid resolution)."""

    data = base_scaling_data
    params = data["params"]

    # Calculate grid points for desired resolution
    x_extent = data["x_range"][-1] - data["x_range"][0]
    z_extent = data["z_range"][-1] - data["z_range"][0]

    n_x = int(x_extent / grid_resolution) + 1
    n_z = int(z_extent / grid_resolution) + 1

    # No grid size limit - let it scale to test performance properly

    # Create new grid
    x_range = np.linspace(data["x_range"][0], data["x_range"][-1], num=n_x, endpoint=True)
    z_range = np.linspace(data["z_range"][0], data["z_range"][-1], num=n_z, endpoint=True)

    x_grid, z_grid = np.meshgrid(x_range, z_range, indexing="ij")
    y_grid = np.zeros_like(x_grid)

    grid_points = np.stack([x_grid.flat, y_grid.flat, z_grid.flat], axis=-1)

    # Compute transmit arrivals for new grid
    sound_speed_m_s = float(params["c"])
    direction = np.asarray(geometry.ultrasound_angles_to_cartesian(0, 0))

    transmit_arrivals_s = (
        wavefront.plane(
            origin_m=np.array([0, 0, 0]),
            points_m=grid_points,
            direction=direction,
        )
        / sound_speed_m_s
    )

    # Transfer to GPU
    iq_data_gpu = cp.asarray(data["iq_data"])
    element_positions_gpu = cp.asarray(data["element_positions"])
    scan_coords_gpu = cp.asarray(grid_points, dtype=cp.float32)
    transmit_arrivals_gpu = cp.asarray(transmit_arrivals_s.flatten(), dtype=cp.float32)

    n_scan = scan_coords_gpu.shape[0]
    n_frames = iq_data_gpu.shape[2]
    out = cp.empty((n_scan, n_frames), dtype=cp.complex64)

    def mach_voxel_scaling():
        """mach GPU function for voxel scaling benchmark."""
        out[:] = 0.0
        result = nb_beamform(
            channel_data=iq_data_gpu,
            rx_coords_m=element_positions_gpu,
            scan_coords_m=scan_coords_gpu,
            tx_wave_arrivals_s=transmit_arrivals_gpu,
            out=out,
            f_number=float(params["fnumber"]),
            rx_start_s=float(params["t0"]),
            sampling_freq_hz=float(params["fs"]),
            sound_speed_m_s=sound_speed_m_s,
            modulation_freq_hz=float(params["fc"]),
            tukey_alpha=0.0,
        )
        return result

    # Benchmark the function
    benchmark(mach_voxel_scaling)

    # Verify basic properties (use the output array, not benchmark return value)
    expected_shape = (n_scan, n_frames)
    assert out.shape == expected_shape
    assert np.isfinite(cp.asnumpy(out)).all()


@pytest.mark.benchmark(
    group="scaling_elements",
    min_time=0.1,
    max_time=1.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.parametrize(
    "element_multiplier",
    [
        pytest.param(1, id="1x_elements"),
        pytest.param(2, id="2x_elements"),
        pytest.param(4, id="4x_elements"),
        pytest.param(8, id="8x_elements"),
        pytest.param(16, id="16x_elements"),
        pytest.param(32, id="32x_elements"),
        pytest.param(64, id="64x_elements"),
    ],
)
def test_scaling_receive_elements(benchmark, base_scaling_data, element_multiplier):
    """Test performance scaling with number of receive elements."""

    data = base_scaling_data
    params = data["params"]

    # Duplicate the element data and positions
    original_iq = data["iq_data"]
    original_positions = data["element_positions"]

    # Tile the IQ data along the element axis
    scaled_iq = np.tile(original_iq, (element_multiplier, 1, 1))

    # Create new element positions by duplicating and slightly offsetting
    n_original_elements = original_positions.shape[0]
    scaled_positions = np.zeros((n_original_elements * element_multiplier, 3), dtype=np.float32)

    for i in range(element_multiplier):
        start_idx = i * n_original_elements
        end_idx = (i + 1) * n_original_elements
        scaled_positions[start_idx:end_idx] = original_positions.copy()
        # Keep identical positions - no offset needed

    # Transfer to GPU
    iq_data_gpu = cp.asarray(scaled_iq)
    element_positions_gpu = cp.asarray(scaled_positions)
    scan_coords_gpu = cp.asarray(data["scan_coords_m"])
    transmit_arrivals_gpu = cp.asarray(data["transmit_arrivals_s"])

    n_scan = scan_coords_gpu.shape[0]
    n_frames = iq_data_gpu.shape[2]
    out = cp.empty((n_scan, n_frames), dtype=cp.complex64)

    def mach_element_scaling():
        """mach GPU function for element scaling benchmark."""
        out[:] = 0.0
        result = nb_beamform(
            channel_data=iq_data_gpu,
            rx_coords_m=element_positions_gpu,
            scan_coords_m=scan_coords_gpu,
            tx_wave_arrivals_s=transmit_arrivals_gpu,
            out=out,
            f_number=float(params["fnumber"]),
            rx_start_s=float(params["t0"]),
            sampling_freq_hz=float(params["fs"]),
            sound_speed_m_s=float(params["c"]),
            modulation_freq_hz=float(params["fc"]),
            tukey_alpha=0.0,
        )
        return result

    # Benchmark the function
    benchmark(mach_element_scaling)

    # Verify basic properties (use the output array, not benchmark return value)
    expected_shape = (n_scan, n_frames)
    assert out.shape == expected_shape
    assert np.isfinite(cp.asnumpy(out)).all()


@pytest.mark.benchmark(
    group="scaling_frames",
    min_time=0.1,
    max_time=1.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.parametrize(
    "frame_multiplier",
    [
        pytest.param(1 / 32, id="1/32x_frames (1 frame)"),
        pytest.param(1 / 8, id="1/8x_frames (4 frames)"),
        pytest.param(1, id="1x_frames"),
        pytest.param(4, id="4x_frames"),
        pytest.param(16, id="16x_frames"),
        pytest.param(64, id="64x_frames"),
    ],
)
def test_scaling_ensemble_size(benchmark, base_scaling_data, frame_multiplier):
    """Test performance scaling with ensemble size (number of frames)."""

    data = base_scaling_data
    params = data["params"]

    # Duplicate the frame data
    original_iq = data["iq_data"]
    if frame_multiplier < 1:
        n_frames = round(original_iq.shape[2] * frame_multiplier)
        scaled_iq = original_iq[:, :, :n_frames]
    else:
        scaled_iq = np.tile(original_iq, (1, 1, frame_multiplier))

    # Transfer to GPU
    iq_data_gpu = cp.asarray(scaled_iq)
    element_positions_gpu = cp.asarray(data["element_positions"])
    scan_coords_gpu = cp.asarray(data["scan_coords_m"])
    transmit_arrivals_gpu = cp.asarray(data["transmit_arrivals_s"])

    n_scan = scan_coords_gpu.shape[0]
    n_frames = iq_data_gpu.shape[2]
    out = cp.empty((n_scan, n_frames), dtype=cp.complex64)

    def mach_frame_scaling():
        """mach GPU function for frame scaling benchmark."""
        out[:] = 0.0
        result = nb_beamform(
            channel_data=iq_data_gpu,
            rx_coords_m=element_positions_gpu,
            scan_coords_m=scan_coords_gpu,
            tx_wave_arrivals_s=transmit_arrivals_gpu,
            out=out,
            f_number=float(params["fnumber"]),
            rx_start_s=float(params["t0"]),
            sampling_freq_hz=float(params["fs"]),
            sound_speed_m_s=float(params["c"]),
            modulation_freq_hz=float(params["fc"]),
            tukey_alpha=0.0,
        )
        return result

    # Benchmark the function
    benchmark(mach_frame_scaling)

    # Verify basic properties (use the output array, not benchmark return value)
    expected_shape = (n_scan, n_frames)
    assert out.shape == expected_shape
    assert np.isfinite(cp.asnumpy(out)).all()
