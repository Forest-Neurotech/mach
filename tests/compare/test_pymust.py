"""Test that mach matches PyMUST, and benchmark performance.

Usage:
    # Run correctness tests only (no benchmarking)
    pytest tests/test_pymust.py --benchmark-disable

    # Benchmark and generate histogram
    pytest tests/test_pymust.py --benchmark-histogram --benchmark-sort=mean
"""

from typing import Any, Callable

import numpy as np
import pytest
import scipy.sparse
from einops import rearrange

pytest.importorskip("pymust")

import pymust

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import jax
    from jax.experimental import sparse as jax_experimental_sparse

    HAS_JAX = True
except ImportError:
    jax = None
    jax_experimental_sparse = None
    HAS_JAX = False

from mach import geometry, wavefront
from mach._vis import save_debug_figures
from mach.kernel import beamform, nb_beamform

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def pymust_iq_data(pymust_data, pymust_params):
    """Extract RF/IQ data from PyMUST data file."""
    mat_data = pymust_data
    rf_data = mat_data["RF"].astype(float)
    iq_data = pymust.rf2iq(rf_data, pymust_params)
    return np.ascontiguousarray(iq_data, dtype=np.complex64)


@pytest.fixture(scope="session")
def pymust_meshgrid(pymust_grid):
    """Generate imaging grid similar to PyMUST example."""
    x_range, y_range, z_range = pymust_grid

    if len(y_range) != 1:
        raise NotImplementedError("PyMUST meshgrid is not implemented for 3D")

    x_grid, z_grid = np.meshgrid(x_range, z_range, indexing="ij")
    y_grid = np.zeros_like(x_grid)

    return (x_grid, y_grid, z_grid), (len(x_range), len(y_range), len(z_range))


@pytest.fixture(scope="session")
def transmit_arrivals_s(pymust_meshgrid, pymust_params):
    """Generate transmit delay matrix."""
    (x_grid, y_grid, z_grid), grid_shape = pymust_meshgrid
    grid_points = np.stack([x_grid.flat, y_grid.flat, z_grid.flat], axis=-1)

    sound_speed_m_s = pymust_params["c"]

    np.testing.assert_allclose(
        actual=np.diff(pymust_params["TXdelay"]),
        desired=0,
        err_msg="Assuming that TXdelay is a 0-degree plane wave, but it is not",
    )
    direction = np.asarray(geometry.ultrasound_angles_to_cartesian(0, 0))

    delays = (
        wavefront.plane(
            origin_m=np.array([0, 0, 0]),
            points_m=grid_points,
            direction=direction,
        )
        / sound_speed_m_s
    )

    return delays


@pytest.fixture(scope="session")
def pymust_das_matrix(pymust_iq_data, pymust_meshgrid, pymust_params):
    """Pre-compute PyMUST DAS matrix for benchmarking."""
    (x_grid, y_grid, z_grid), _ = pymust_meshgrid

    # Create DAS matrix once (it's the same for all frames)
    M = pymust.dasmtx(1j * np.array(pymust_iq_data.shape[:2]), x_grid, z_grid, pymust_params)
    return M


@pytest.fixture(scope="session")
def benchmark_data(pymust_iq_data, pymust_element_positions, pymust_meshgrid, transmit_arrivals_s, pymust_params):
    """Pre-load all data for benchmarking, avoiding data transfer overhead."""
    (x_grid, y_grid, z_grid), grid_shape = pymust_meshgrid

    # Prepare data for CUDA beamformer
    iq_data_reordered = np.ascontiguousarray(
        rearrange(pymust_iq_data, "n_samples n_elements n_frames -> n_elements n_samples n_frames"), dtype=np.complex64
    )

    # GPU data (if available)
    gpu_data = {}
    if HAS_CUPY:
        iq_data_gpu = cp.zeros_like(iq_data_reordered)
        cp.cuda.runtime.memcpy(
            iq_data_gpu.data.ptr,
            iq_data_reordered.ctypes.data,
            iq_data_reordered.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
        )

        gpu_data.update({
            "iq_data_gpu": iq_data_gpu,
            "element_positions_gpu": cp.asarray(pymust_element_positions, dtype=cp.float32),
            "scan_coords_m_gpu": cp.asarray(
                np.stack([x_grid.flat, y_grid.flat, z_grid.flat], axis=-1), dtype=cp.float32
            ),
            "transmit_arrivals_s_gpu": cp.asarray(transmit_arrivals_s.flatten(), dtype=cp.float32),
        })

    return {
        # CPU data
        "pymust_iq_data": pymust_iq_data,
        "iq_data_cpu": iq_data_reordered,
        "element_positions_cpu": pymust_element_positions,
        "scan_coords_m_cpu": np.stack([x_grid.flat, y_grid.flat, z_grid.flat], axis=-1),
        "transmit_arrivals_s_cpu": transmit_arrivals_s.flatten(),
        # GPU data (if available)
        **gpu_data,
        # Grid info
        "x_grid": x_grid,
        "z_grid": z_grid,
        "grid_shape": grid_shape,
        # Parameters
        "params": pymust_params,
    }


# ============================================================================
# Helper Functions
# ============================================================================


# ============================================================================
# Unified Tests (Correctness + Benchmarking)
# ============================================================================


@pytest.fixture(
    params=[
        pytest.param(scipy.sparse.coo_matrix, id="cpu"),
        pytest.param(
            jax_experimental_sparse.BCOO.from_scipy_sparse,
            id="gpu",
            marks=[
                pytest.mark.skip(reason="jax.experimental.sparse.BCOO is experimental"),
                pytest.mark.skipif(not HAS_JAX, reason="JAX not available"),
            ],
        ),
    ]
)
def sparse_backend(request):
    """Fixture providing different sparse-matrix backends."""
    return request.param


@pytest.mark.benchmark(
    group="doppler_disk",
    min_time=0.1,
    max_time=5.0,
    min_rounds=5,
    warmup=True,
    warmup_iterations=1,
)
def test_pymust_benchmark(
    benchmark, benchmark_data, pymust_das_matrix, output_dir, sparse_backend: Callable[[scipy.sparse.coo_matrix], Any]
):
    """
    Unified test: PyMUST CPU implementation vs mach.

    - With --benchmark-disable: Tests correctness only, without benchmarking
    """
    data = benchmark_data
    M = pymust_das_matrix
    iq_data = data["pymust_iq_data"]
    x_grid = data["x_grid"]

    # Flatten data for PyMUST
    flattened_data = iq_data.reshape(-1, iq_data.shape[2], order="F")

    USE_JAX_SPARSE = HAS_JAX and (sparse_backend == jax_experimental_sparse.BCOO.from_scipy_sparse)
    if USE_JAX_SPARSE:
        device = jax.devices("gpu")[0]
        M = jax_experimental_sparse.BCOO.from_scipy_sparse(M)
        M = jax.device_put(M, device)
        flattened_data = jax.device_put(flattened_data, device)

        @jax.jit
        def beamform_jax_jit(M, flattened_data):
            """Beamform function to benchmark."""
            result = M @ flattened_data
            return jax.block_until_ready(result)

        def pymust_beamform():
            """PyMUST beamforming function to benchmark."""
            result = beamform_jax_jit(M, flattened_data)
            return result.block_until_ready()

    else:

        def pymust_beamform():
            """PyMUST beamforming function to benchmark."""
            result = M @ flattened_data
            return result

    # Run PyMUST with benchmark timing (or just run normally if benchmarking disabled)
    pymust_result_flat = benchmark(pymust_beamform)

    if USE_JAX_SPARSE:
        pymust_result_flat = np.asarray(pymust_result_flat)

    # Reshape PyMUST result to match expected output format
    pymust_result = pymust_result_flat.reshape(x_grid.shape[0], x_grid.shape[1], iq_data.shape[2], order="F")
    assert np.isfinite(pymust_result).all()


@pytest.mark.benchmark(
    group="doppler_disk",
    min_time=0.1,
    max_time=5.0,
    min_rounds=5,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
def test_mach(benchmark, benchmark_data, pymust_das_matrix, output_dir):
    """
    Unified test: mach GPU implementation.

    - With --benchmark-disable: Tests correctness against PyMUST
    - With --benchmark-only: Benchmarks GPU performance only
    - With both enabled: Tests correctness AND benchmarks performance
    """
    data = benchmark_data
    params = data["params"].copy()
    sampling_freq_hz = float(params["fs"])
    modulation_freq_hz = float(params["fc"])
    sound_speed_m_s = float(params["c"])
    rx_start_s = float(params["t0"])
    f_number = float(params["fnumber"])

    n_scan = data["scan_coords_m_gpu"].shape[0]
    n_frames = data["iq_data_gpu"].shape[2]
    out = cp.empty((n_scan, n_frames), dtype=cp.complex64)

    def mach_gpu():
        """mach GPU function to benchmark."""
        # Clear output array before each benchmark run
        out[:] = 0.0
        result = nb_beamform(
            channel_data=data["iq_data_gpu"],
            rx_coords_m=data["element_positions_gpu"],
            scan_coords_m=data["scan_coords_m_gpu"],
            tx_wave_arrivals_s=data["transmit_arrivals_s_gpu"],
            out=out,
            f_number=f_number,
            rx_start_s=rx_start_s,
            sampling_freq_hz=sampling_freq_hz,
            sound_speed_m_s=sound_speed_m_s,
            modulation_freq_hz=modulation_freq_hz,
            tukey_alpha=0.0,
        )
        return result

    # Run GPU beamforming with benchmark timing (or just run normally if benchmarking disabled)
    benchmark(mach_gpu)
    gpu_result = out

    # Verify basic properties
    expected_shape = (data["scan_coords_m_gpu"].shape[0], data["iq_data_gpu"].shape[2])
    assert gpu_result.shape == expected_shape

    gpu_result_cpu = cp.asnumpy(gpu_result)
    assert np.isfinite(gpu_result_cpu).all()

    # For correctness comparison, run PyMUST reference (not benchmarked)
    M = pymust_das_matrix
    iq_data = data["pymust_iq_data"]
    flattened_data = iq_data.reshape(-1, iq_data.shape[2], order="F")
    pymust_result_flat = M @ flattened_data
    pymust_result = pymust_result_flat.reshape(
        data["x_grid"].shape[0], data["x_grid"].shape[1], iq_data.shape[2], order="F"
    )

    # Take power-doppler for comparison
    pymust_power_doppler = np.square(np.abs(pymust_result)).sum(axis=-1)
    our_power_doppler = np.square(np.abs(gpu_result_cpu)).sum(axis=-1)

    # Use 2D grid shape
    grid_shape = data["x_grid"].shape

    # Save debug output if requested
    if output_dir is not None:
        save_debug_figures(
            our_result=our_power_doppler,
            reference_result=pymust_power_doppler,
            grid_shape=grid_shape,
            x_axis=data["x_grid"],
            z_axis=data["z_grid"],
            output_dir=output_dir / "pymust_comparison",
            test_name="mach_gpu_comparison",
            our_label="mach",
            reference_label="PyMUST",
        )

    # Correctness check
    np.testing.assert_allclose(
        actual=our_power_doppler.reshape(grid_shape),
        desired=pymust_power_doppler,
        atol=1,
        rtol=1e-3,
        err_msg="GPU beamforming results do not match PyMUST",
    )


@pytest.mark.benchmark(
    group="load_from_cpu: doppler_disk",
    min_time=0.1,
    max_time=5.0,
    min_rounds=5,
    warmup=True,
    warmup_iterations=1,
)
def test_mach_from_cpu(benchmark, benchmark_data, pymust_das_matrix, output_dir):
    """Test mach loading data from the CPU.

    This adds the overhead of transferring data from the CPU to the GPU.
    """
    data = benchmark_data
    params = data["params"].copy()
    sampling_freq_hz = float(params["fs"])
    modulation_freq_hz = float(params["fc"])
    sound_speed_m_s = float(params["c"])
    rx_start_s = float(params["t0"])
    f_number = float(params["fnumber"])

    n_scan = data["scan_coords_m_cpu"].shape[0]
    n_frames = data["iq_data_cpu"].shape[2]
    out = np.empty((n_scan, n_frames), dtype=np.complex64)

    def mach_cpu():
        """mach CPU function to benchmark."""
        result = beamform(
            channel_data=data["iq_data_cpu"],
            rx_coords_m=data["element_positions_cpu"],
            scan_coords_m=data["scan_coords_m_cpu"],
            tx_wave_arrivals_s=data["transmit_arrivals_s_cpu"],
            out=out,
            f_number=f_number,
            rx_start_s=rx_start_s,
            sampling_freq_hz=sampling_freq_hz,
            sound_speed_m_s=sound_speed_m_s,
            modulation_freq_hz=modulation_freq_hz,
            tukey_alpha=0.0,
        )
        return result

    # Run CPU beamforming with benchmark timing (or just run normally if benchmarking disabled)
    cpu_result = benchmark(mach_cpu)

    # Verify basic properties
    expected_shape = (data["scan_coords_m_cpu"].shape[0], data["iq_data_cpu"].shape[2])
    assert cpu_result.shape == expected_shape
    assert np.isfinite(cpu_result).all()

    # For correctness comparison, run PyMUST reference (not benchmarked)
    M = pymust_das_matrix
    iq_data = data["pymust_iq_data"]
    flattened_data = iq_data.reshape(-1, iq_data.shape[2], order="F")
    pymust_result_flat = M @ flattened_data
    pymust_result = pymust_result_flat.reshape(
        data["x_grid"].shape[0], data["x_grid"].shape[1], iq_data.shape[2], order="F"
    )

    # Take power-doppler for comparison
    pymust_power_doppler = np.square(np.abs(pymust_result)).sum(axis=-1)
    our_power_doppler = np.square(np.abs(cpu_result)).sum(axis=-1)

    # Use 2D grid shape
    grid_shape = data["x_grid"].shape

    # Save debug output if requested
    if output_dir is not None:
        save_debug_figures(
            our_result=our_power_doppler,
            reference_result=pymust_power_doppler,
            grid_shape=grid_shape,
            x_axis=data["x_grid"],
            z_axis=data["z_grid"],
            output_dir=output_dir / "pymust_comparison",
            test_name="mach_cpu_comparison",
            our_label="mach (CPU→GPU)",
            reference_label="PyMUST",
        )

    # Correctness check (always performed unless --benchmark-only)
    np.testing.assert_allclose(
        actual=our_power_doppler.reshape(grid_shape),
        desired=pymust_power_doppler,
        atol=1,
        rtol=1e-3,
        err_msg="CPU beamforming results do not match PyMUST",
    )


if __name__ == "__main__":
    import pytest

    pytest.main([
        __file__,
        "-v",
        "--benchmark-histogram",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops,rounds,iterations",
        "-W",
        "ignore::pytest.PytestAssertRewriteWarning",
    ])
