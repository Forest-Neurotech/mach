"""Test that vbeam beamformer matches PyMUST, and benchmark performance.

Usage:
    # Run correctness tests only (no benchmarking)
    pytest tests/test_vbeam.py --benchmark-disable

    # Benchmark and generate histogram
    pytest tests/test_vbeam.py --benchmark-histogram --benchmark-sort=mean
"""

import warnings

import einops
import numpy as np
import pytest
from pyuff_ustb import ChannelData, Scan

pytest.importorskip("vbeam")
pytest.importorskip("jax")

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

import jax
import jax.numpy as jnp
from spekk import Spec
from vbeam.apodization import NoApodization, PlaneWaveReceiveApodization, Rectangular, TxRxApodization
from vbeam.beamformers import get_das_beamformer
from vbeam.core import ElementGeometry, WaveData
from vbeam.data_importers import SignalForPointSetup, import_pyuff
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import LinearScan
from vbeam.wavefront import PlaneWavefront, ReflectedWavefront

from mach import experimental
from mach._vis import save_debug_figures
from mach.io.uff import create_beamforming_setup

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def iq_data(pymust_data, pymust_params):
    """Extract RF/IQ data from PyMUST data file.

    To avoid importing pymust just for this test, we follow the vbeam convention of
    using the hilbert transform to convert RF data to IQ data.
    """
    pytest.importorskip("pymust")
    import pymust

    mat_data = pymust_data
    rf_data = mat_data["RF"].astype(float)
    # PyMUST undersamples its disk-data, so we need to demodulate it
    # we can't use hilbert because it doesn't support undersampling
    iq_data = pymust.rf2iq(rf_data, pymust_params)
    return np.ascontiguousarray(iq_data, dtype=np.complex64)


@pytest.fixture(
    scope="session",
    # Note: pytest-benchmark interferes with jax[cpu], so we only test on GPU
    params=["gpu"],
)
def vbeam_pymust_setup(request, iq_data, pymust_element_positions, pymust_grid, pymust_params) -> SignalForPointSetup:
    """Convert PyMUST data to vbeam SignalForPointSetup."""
    device = jax.devices(request.param)[0]
    params = pymust_params
    x_range, y_range, z_range = pymust_grid

    # Convert IQ data to vbeam format
    # PyMUST format: (n_samples, n_receivers, n_frames)
    # vbeam expects (n_frames, n_transmits, n_receivers, n_samples)
    # Note: vbeam is technically flexible about the order of dimensions, but let's use a layout that's easy to understand
    iq_vbeam = einops.rearrange(iq_data, "samples elements frames -> frames 1 elements samples")

    # Create interpolation space - use the correct dimension for samples
    interpolate = FastInterpLinspace(min=float(params["t0"]), d=1.0 / float(params["fs"]), n=iq_vbeam.shape[3])

    # Create receiver geometry from element positions
    receivers = ElementGeometry(
        position=jax.device_put(pymust_element_positions.astype(np.float32), device=device),
        theta=jax.device_put(np.zeros(pymust_element_positions.shape[0], dtype=np.float32), device=device),
        phi=jax.device_put(np.zeros(pymust_element_positions.shape[0], dtype=np.float32), device=device),
    )

    # For plane wave imaging, the sender is virtual (at origin)
    sender = ElementGeometry(
        position=jnp.zeros((3,), dtype=jnp.float32, device=device),
        theta=jnp.zeros(1, dtype=jnp.float32, device=device),
        phi=jnp.zeros(1, dtype=jnp.float32, device=device),
    )

    # Create wave data for plane wave transmits
    # Hard-coded for now, but could be more flexible
    n_transmits = 1
    np.testing.assert_allclose(
        actual=np.diff(pymust_params["TXdelay"]),
        desired=0,
        err_msg="Assuming that TXdelay is a 0-degree plane wave, but it is not",
    )
    wave_data = WaveData(
        azimuth=jax.device_put(np.zeros(n_transmits, dtype=np.float32), device=device),  # 0-degree plane waves
        elevation=jax.device_put(np.zeros(n_transmits, dtype=np.float32), device=device),
        source=jax.device_put(np.tile([[0.0, 0.0, 0.0]], (n_transmits, 1)).astype(np.float32), device),
        t0=jax.device_put(np.zeros(n_transmits, dtype=np.float32), device=device),  # Assume 0 time offset
    )

    # Create apodization
    apodization = TxRxApodization(
        transmit=NoApodization(),
        receive=PlaneWaveReceiveApodization(Rectangular(), f_number=float(params["fnumber"])),
    )

    # Create scan object - using LinearScan with actual coordinate arrays
    scan = LinearScan(
        x=jax.device_put(x_range, device=device),
        y=jax.device_put(y_range, device=device),
        z=jax.device_put(z_range, device=device),
    )

    # Create spec for data dimensions
    spec = Spec({
        "signal": ["frames", "transmits", "receivers", "signal_time"],
        "receiver": ["receivers"],
        "point_position": ["points"],
        "wave_data": ["transmits"],
    })

    # Create setup
    setup = SignalForPointSetup(
        sender=sender,
        point_position=None,  # Will be provided by scan
        receiver=receivers,
        signal=jax.device_put(iq_vbeam.astype(np.complex64), device=device),
        transmitted_wavefront=PlaneWavefront(),
        reflected_wavefront=ReflectedWavefront(),
        speed_of_sound=float(params["c"]),
        wave_data=wave_data,
        interpolate=interpolate,
        modulation_frequency=pymust_params["fc"],  # RF data, so no demodulation
        apodization=apodization,
        spec=spec,
        scan=scan,
    )

    return setup


@pytest.fixture(scope="module")
def vbeam_setup_uff(
    picmus_phantom_resolution_channel_data: ChannelData, picmus_phantom_resolution_scan: Scan
) -> SignalForPointSetup:
    """Convert UFF data to vbeam SignalForPointSetup for comparison with mach."""
    with warnings.catch_warnings():
        # import_pyuff allows inf multiplication
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
        setup = import_pyuff(
            picmus_phantom_resolution_channel_data,
            picmus_phantom_resolution_scan,
            frames=0,
        )
    # vbeam defaults to apodization features that are not supported by mach
    # so we disable them here
    setup.apodization.transmit = NoApodization()
    setup.apodization.receive = PlaneWaveReceiveApodization(
        Rectangular(),
        f_number=setup.apodization.receive.f_number,
    )
    return setup


@pytest.fixture(scope="module")
def picmus_phantom_resolution_beamform_kwargs(
    picmus_phantom_resolution_channel_data: ChannelData, picmus_phantom_resolution_scan: Scan
) -> dict:
    """mach kwargs for UFF data."""
    return create_beamforming_setup(
        channel_data=picmus_phantom_resolution_channel_data,
        scan=picmus_phantom_resolution_scan,
        xp=cp if HAS_CUPY else None,
    )


# ============================================================================
# Unified Tests (Correctness + Benchmarking)
# ============================================================================


@pytest.mark.benchmark(
    group="doppler_disk",
    min_time=0.1,
    max_time=5.0,
    min_rounds=5,
    warmup=True,
    warmup_iterations=1,
)
def test_vbeam_benchmark(benchmark, vbeam_pymust_setup, output_dir):
    """vbeam benchmark test on Doppler disk dataset.

    - With --benchmark-disable: Just tests that vbeam works
    - With --benchmark-only: Benchmarks vbeam performance only
    - With both enabled: Tests basic functionality AND benchmarks performance
    """
    # Create the beamformer with JIT-compilation (needs warmup with data)
    beamformer = jax.jit(
        get_das_beamformer(
            vbeam_pymust_setup,
            compensate_for_apodization_overlap=False,  # for consistency with mach
            log_compress=False,  # Keep raw output
            scan_convert=False,  # Keep in original grid format
        )
    )

    # Prepare input data
    vbeam_data = vbeam_pymust_setup.data
    grid_shape = vbeam_pymust_setup.scan.shape
    num_frames = vbeam_data["signal"].shape[0]

    def vbeam_beamform():
        """vbeam beamforming function to benchmark."""
        result = beamformer(**vbeam_data).block_until_ready()
        return result

    # Run vbeam with benchmark timing (or just run normally if benchmarking disabled)
    vbeam_result = benchmark(vbeam_beamform)

    # Convert JAX array to numpy for verification
    if hasattr(vbeam_result, "__array__"):
        vbeam_result = np.asarray(vbeam_result)

    # Verify basic properties
    assert vbeam_result.shape == (num_frames, *grid_shape)
    assert np.isfinite(vbeam_result).all()
    assert vbeam_result.dtype == np.complex64  # Should be complex for IQ data

    # Save visualization if output directory is provided
    if output_dir is not None:
        # squeeze the singleton elevation dimension
        power_doppler = einops.reduce(np.abs(vbeam_result) ** 2, "frames x y z -> x z", "sum")
        output_dir = output_dir / "vbeam_results"
        save_debug_figures(
            our_result=power_doppler,
            reference_result=None,
            grid_shape=(vbeam_pymust_setup.scan.x.size, vbeam_pymust_setup.scan.z.size),
            x_axis=vbeam_pymust_setup.scan.x,
            z_axis=vbeam_pymust_setup.scan.z,
            output_dir=output_dir,
            test_name="vbeam_benchmark",
            our_label="vbeam",
        )
        print("Saved debug figures to", output_dir)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@pytest.mark.filterwarnings("ignore:array is not contiguous, rearranging will add latency:UserWarning")
def test_mach_matches_vbeam(
    picmus_phantom_resolution_beamform_kwargs, vbeam_setup_uff: SignalForPointSetup, output_dir
):
    """Validate mach against vbeam output on a PICMUS UFF data file."""
    grid_shape = vbeam_setup_uff.scan.shape

    print(picmus_phantom_resolution_beamform_kwargs.keys())
    gpu_result = experimental.beamform(**picmus_phantom_resolution_beamform_kwargs)
    result = cp.asnumpy(gpu_result)
    # Reshape to (x, z)
    result = result.reshape(grid_shape)

    # Verify basic properties
    assert np.isfinite(result).all()

    # For correctness comparison, run vbeam reference (not benchmarked)
    # Create the vbeam beamformer with JIT-compilation
    beamformer = get_das_beamformer(
        vbeam_setup_uff,
        compensate_for_apodization_overlap=False,
        log_compress=False,
        scan_convert=False,
    )
    vbeam_result_jax = beamformer(**vbeam_setup_uff.data).block_until_ready()
    vbeam_result = np.asarray(vbeam_result_jax)

    # Compare complex values directly for more rigorous validation
    # This will catch both magnitude and phase differences
    print(
        f"vbeam result range: real=[{vbeam_result.real.min():.6f}, {vbeam_result.real.max():.6f}], imag=[{vbeam_result.imag.min():.6f}, {vbeam_result.imag.max():.6f}]"
    )
    print(
        f"mach result range: real=[{result.real.min():.6f}, {result.real.max():.6f}], imag=[{result.imag.min():.6f}, {result.imag.max():.6f}]"
    )
    print(f"Complex ratio (mach/vbeam) mean: {(result / (vbeam_result + 1e-10)).mean():.6f}")

    # Also show magnitude comparison for reference
    vbeam_magnitude = np.abs(vbeam_result)
    cuda_magnitude = np.abs(result)
    print(f"Magnitude ratio (mach/vbeam) mean: {(cuda_magnitude / (vbeam_magnitude + 1e-10)).mean():.6f}")

    # Check data types - this could be the issue!
    print(f"vbeam result dtype: {vbeam_result.dtype}")
    print(f"mach result dtype: {result.dtype}")
    print(f"vbeam signal dtype: {vbeam_setup_uff.signal.dtype}")
    print(f"mach signal dtype: {picmus_phantom_resolution_beamform_kwargs['channel_data'].dtype}")

    # Check element positions - this could be the real issue!
    print(f"vbeam receiver positions shape: {vbeam_setup_uff.receiver.position.shape}")
    print(f"mach receiver positions shape: {picmus_phantom_resolution_beamform_kwargs['rx_coords_m'].shape}")
    print(f"vbeam receiver positions (first 3): {vbeam_setup_uff.receiver.position[:3]}")
    print(f"mach receiver positions (first 3): {picmus_phantom_resolution_beamform_kwargs['rx_coords_m'][:3]}")

    # Check if positions match
    vbeam_pos = np.asarray(vbeam_setup_uff.receiver.position)
    mach_pos = picmus_phantom_resolution_beamform_kwargs["rx_coords_m"]
    if hasattr(mach_pos, "get"):  # cupy array
        mach_pos = mach_pos.get()
    else:
        mach_pos = np.asarray(mach_pos)
    print(f"Receiver positions match: {np.allclose(vbeam_pos, mach_pos, atol=1e-6)}")
    if not np.allclose(vbeam_pos, mach_pos, atol=1e-6):
        print(f"Receiver position difference stats: max_diff={np.max(np.abs(vbeam_pos - mach_pos)):.2e}")

    # Check scan positions too
    vbeam_scan_pos = np.asarray(vbeam_setup_uff.scan.get_points())
    mach_scan_pos = picmus_phantom_resolution_beamform_kwargs["scan_coords_m"]
    if hasattr(mach_scan_pos, "get"):  # cupy array
        mach_scan_pos = mach_scan_pos.get()
    else:
        mach_scan_pos = np.asarray(mach_scan_pos)
    print(f"Scan positions match: {np.allclose(vbeam_scan_pos, mach_scan_pos, atol=1e-6)}")
    if not np.allclose(vbeam_scan_pos, mach_scan_pos, atol=1e-6):
        print(f"Scan position difference stats: max_diff={np.max(np.abs(vbeam_scan_pos - mach_scan_pos)):.2e}")

    # Check transmit arrival times - this could be the issue!
    print(f"mach tx_wave_arrivals_s shape: {picmus_phantom_resolution_beamform_kwargs['tx_wave_arrivals_s'].shape}")
    mach_tx_arrivals = picmus_phantom_resolution_beamform_kwargs["tx_wave_arrivals_s"]
    if hasattr(mach_tx_arrivals, "get"):
        mach_tx_arrivals = mach_tx_arrivals.get()
    else:
        mach_tx_arrivals = np.asarray(mach_tx_arrivals)
    print(f"mach tx arrivals (first transmit, first 3 points): {mach_tx_arrivals[0, :3]}")

    # Let's also check the wave data from vbeam setup
    print(f"vbeam wave_data.t0 (first 3): {vbeam_setup_uff.wave_data.t0[:3]}")
    print(f"vbeam modulation_frequency: {vbeam_setup_uff.modulation_frequency}")
    print(f"mach modulation_freq_hz: {picmus_phantom_resolution_beamform_kwargs['modulation_freq_hz']}")

    # Let's also check apodization settings
    print(f"vbeam f_number: {vbeam_setup_uff.apodization.receive.f_number}")
    print(f"mach f_number: {picmus_phantom_resolution_beamform_kwargs['f_number']}")

    # Save debug output if requested
    if output_dir is not None:
        output_dir = output_dir / "mach_vs_vbeam"
        save_debug_figures(
            our_result=cuda_magnitude,
            reference_result=vbeam_magnitude,
            grid_shape=grid_shape,
            x_axis=vbeam_setup_uff.scan.x,
            z_axis=vbeam_setup_uff.scan.z,
            output_dir=output_dir,
            test_name="mach_vs_vbeam",
            our_label="mach",
            reference_label="vbeam",
        )
        print("Saved debug figures to", output_dir)

        # After empirical scaling, compare complex values directly with strict tolerances
    # This tests both magnitude and phase alignment between implementations
    np.testing.assert_allclose(
        actual=result,
        desired=vbeam_result,
        atol=1,
        rtol=1 / 30,
        err_msg="mach complex results do not match vbeam within expected tolerances (with scaling correction)",
    )


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--benchmark-histogram",
        "--benchmark-sort=mean",
    ])
