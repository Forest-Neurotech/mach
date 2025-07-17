"""Python bindings and wrapper for the CUDA kernel."""

from typing import Optional, cast

from array_api_compat import is_writeable_array
from jaxtyping import Num, Real

from mach._array_api import Array, array_namespace
from mach._check import ensure_contiguous, is_contiguous

# Import from the nanobind module
from ._cuda_impl import __nvcc_version__  # type: ignore[attr-defined]  # noqa: F401
from ._cuda_impl import beamform as nb_beamform  # type: ignore[attr-defined]


def beamform(  # noqa: C901
    channel_data: Num[Array, "n_rx n_samples n_frames"],
    rx_coords_m: Real[Array, "n_rx xyz=3"],
    scan_coords_m: Real[Array, "n_scan xyz=3"],
    tx_wave_arrivals_s: Real[Array, " n_scan"],
    out: Optional[Num[Array, "n_scan n_frames"]] = None,
    *,
    rx_start_s: float,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: Optional[float] = None,
    tukey_alpha: float = 0.5,
) -> Array:
    """CUDA ultrasound beamforming with automatic GPU/CPU dispatch.

    This function implements delay-and-sum beamforming with the following features:
    - Dynamic aperture growth based on F-number
    - Tukey apodization with adjustable taper width
    - Support for both RF and IQ data
    - Multi-frame processing
    - Automatic array protocol detection and GPU/CPU dispatch
    - Mixed CPU/GPU array handling with automatic memory management

    For theoretical background on delay-and-sum beamforming, see:
    Perrot et al., "So you think you can DAS? A viewpoint on delay-and-sum beamforming"
    https://www.biomecardio.com/publis/ultrasonics21.pdf

    `beamform` is a wrapper around `nb_beamform`, a nanobind-generated
    Python/C++/CUDA function. `beamform` adds more helpful error-messages than `nb_beamform`,
    but it adds about 0.1ms overhead (on AMD Ryzen Threadripper Pro).
    If your inputs are properly shaped/typed, or you are okay reading nanobind's
    slightly-confusing type-check error messages, you can use `nb_beamform` directly.

    Args:
        channel_data:
            RF/IQ data with shape (n_rx, n_samples, n_frames).
            For I/Q data: use complex64 dtype.
            For RF data: use float32 dtype.
            Note: this layout order improves memory-access patterns for the CUDA kernel.
        rx_coords_m:
            Receive element positions with shape (n_rx, 3) where each row is [x, y, z] in meters.
            Each element represents the physical location of a transducer element on the probe.
        scan_coords_m:
            Scan grid point coordinates with shape (n_scan, 3) where each row is [x, y, z] in meters.
            These are the spatial locations where beamformed values will be computed.
            Note: n_scan = number of points_m in the imaging grid where you want beamformed output.
        tx_wave_arrivals_s:
            Transmit wave arrival times with shape (n_scan,) in seconds.
            This represents the time when the transmitted acoustic wave arrives at each
            scan grid point. For different transmit types:
            - Plane wave: arrivals computed from wave direction and grid positions
            - Focused/diverging wave: arrivals computed from focal point and grid positions

            Use `mach.wavefront.plane() / sound_speed_m_s` or
            `mach.wavefront.spherical() / sound_speed_m_s` to compute these values.
        out:
            Optional output array with shape (n_scan, nframes).
            Must match input type: complex64 for I/Q, float32 for RF.
        rx_start_s:
            Receive start time offset in seconds. This corresponds to t0 in the literature
            (biomecardio.com/publis/ultrasonics21.pdf) - the time when the 0th sample
            was recorded relative to the transmit event. When rx_start_s=0, the wave
            is assumed to pass through the coordinate origin_m at t=0.
        sampling_freq_hz:
            Sampling frequency in Hz.
        f_number:
            F-number for aperture calculations. Controls the size of the receive aperture
            based on depth. Typical values range from 1.0 to 3.0.
        sound_speed_m_s:
            Speed of sound in m/s. Typical value for soft tissue is ~1540 m/s.
        modulation_freq_hz:
            Center frequency in Hz (only used for I/Q data; ignored for RF data).
            For I/Q data: required parameter, set to 0 if no demodulation was used.
            For RF data: automatically defaults to 0.0 if not provided.
        tukey_alpha:
            Tukey window alpha parameter for apodization. Range [0, 1]:
            - 0.0: no apodization (rectangular window)
            - 0.5: moderate apodization (default)
            - 1.0: maximum apodization (Hann window)

    Returns:
        Beamformed data with shape (n_scan, nframes).
        Will be out if provided, otherwise a new array will be created.
        Output dtype matches input dtype (complex64 or float32).

    Notes:
        - All spatial coordinates should be in meters.
        - All time values should be in seconds.
        - All frequencies should be in Hz.
        - For optimal performance, use contiguous arrays with appropriate dtypes.
        - If the input arrays are not contiguous, the function automatically handles memory layout conversion.
        - Arrays can be on different devices (CPU/GPU); automatic copying will be performed with performance warnings.

    Examples:
        Basic plane wave beamforming:

        >>> import numpy as np
        >>> from mach import beamform, wavefront
        >>>
        >>> # Set up geometry
        >>> rx_positions = np.array([[0, 0, 0], [1e-3, 0, 0]])  # 2 elements, 1mm spacing
        >>> scan_points = np.array([[0, 0, 10e-3], [0, 0, 20e-3]])  # 2 depths: 10mm, 20mm
        >>>
        >>> # Compute transmit arrivals for 0° plane wave
        >>> arrivals_dist = wavefront.plane(
        ...     origin_m=np.array([0, 0, 0]),
        ...     points_m=scan_points,
        ...     direction=np.array([0, 0, 1])  # +z direction
        ... )
        >>> tx_arrivals = arrivals_dist / 1540  # Convert to time (assuming 1540 m/s)
        >>>
        >>> # Beamform (assuming you have channel_data)
        >>> result = beamform(
        ...     channel_data=channel_data,  # shape: (2, n_samples, n_frames)
        ...     rx_coords_m=rx_positions,
        ...     scan_coords_m=scan_points,
        ...     tx_wave_arrivals_s=tx_arrivals,
        ...     rx_start_s=0.0,
        ...     sampling_freq_hz=40e6,
        ...     f_number=1.5,
        ...     sound_speed_m_s=1540
        ... )
    """

    # Check input type and dtype before trying to convert dtypes
    # shape should be checked in the kernel
    if not isinstance(channel_data, Num[Array, "..."]):
        channel_data_type = type(channel_data)
        channel_data_dtype = getattr(channel_data_type, "dtype", None)
        raise TypeError(
            f"channel_data must be array with dtype=numeric, got type={channel_data_type}, dtype={channel_data_dtype}"
        )
    if not isinstance(rx_coords_m, Real[Array, "..."]):
        rx_coords_m_type = type(rx_coords_m)
        rx_coords_m_dtype = getattr(rx_coords_m_type, "dtype", None)
        raise TypeError(
            f"rx_coords_m must be array with dtype=Real, got type={rx_coords_m_type}, dtype={rx_coords_m_dtype}"
        )
    if not isinstance(scan_coords_m, Real[Array, "..."]):
        scan_coords_m_type = type(scan_coords_m)
        scan_coords_m_dtype = getattr(scan_coords_m_type, "dtype", None)
        raise TypeError(
            f"scan_coords_m must be array with dtype=Real, got type={scan_coords_m_type}, dtype={scan_coords_m_dtype}"
        )
    if not isinstance(tx_wave_arrivals_s, Real[Array, "..."]):
        tx_wave_arrivals_s_type = type(tx_wave_arrivals_s)
        tx_wave_arrivals_s_dtype = getattr(tx_wave_arrivals_s_type, "dtype", None)
        raise TypeError(
            f"tx_wave_arrivals_s must be array with dtype=Real, got type={tx_wave_arrivals_s_type}, dtype={tx_wave_arrivals_s_dtype}"
        )
    if (out is not None) and (not isinstance(out, Num[Array, "..."])):
        out_type = type(out)
        out_dtype = getattr(out_type, "dtype", None)
        raise TypeError(f"out must be array with dtype=numeric, got type={out_type}, dtype={out_dtype}")

    # Get array namespaces
    xp_data = array_namespace(channel_data)
    xp_coords = array_namespace(rx_coords_m)
    xp_grid = array_namespace(scan_coords_m)
    xp_idt = array_namespace(tx_wave_arrivals_s)

    # Ensure arrays have required DLPack methods
    for arr, name in [
        (channel_data, "channel_data"),
        (rx_coords_m, "rx_coords_m"),
        (scan_coords_m, "scan_coords_m"),
        (tx_wave_arrivals_s, "tx_wave_arrivals_s"),
    ]:
        if not hasattr(arr, "__dlpack_device__"):
            raise TypeError(f"Array '{name}' does not support DLPack protocol")

    if out is not None and not hasattr(out, "__dlpack_device__"):
        raise TypeError("Array 'out' does not support DLPack protocol")

    # Ensure float32 dtype for coordinate arrays
    rx_coords_m = rx_coords_m.astype(xp_coords.float32, copy=False)
    scan_coords_m = scan_coords_m.astype(xp_grid.float32, copy=False)
    tx_wave_arrivals_s = tx_wave_arrivals_s.astype(xp_idt.float32, copy=False)

    nframes = channel_data.shape[2]
    n_scan = scan_coords_m.shape[0]

    # Determine data type and prepare sensor data
    is_complex = (channel_data.dtype == xp_data.complex64) or (channel_data.dtype == xp_data.complex128)
    if is_complex:
        channel_data = channel_data.astype(xp_data.complex64, copy=False)
        output_dtype = xp_data.complex64
        if modulation_freq_hz is None:
            raise ValueError(
                "modulation_freq_hz is required for complex phase-correction. set it to 0 if no demodulation was used."
            )
    else:
        channel_data = channel_data.astype(xp_data.float32, copy=False)
        output_dtype = xp_data.float32
        if modulation_freq_hz is None:
            modulation_freq_hz = 0.0

    # Check for contiguous arrays in libraries that support it
    channel_data = ensure_contiguous(channel_data)
    rx_coords_m = ensure_contiguous(rx_coords_m)
    scan_coords_m = ensure_contiguous(scan_coords_m)
    tx_wave_arrivals_s = ensure_contiguous(tx_wave_arrivals_s)

    if out is None:
        out = xp_data.zeros((n_scan, nframes), dtype=output_dtype)
    else:
        # Validate output array dtype matches input
        expected_dtype = xp_data.complex64 if is_complex else xp_data.float32
        if out.dtype != expected_dtype:
            raise ValueError(f"Output array dtype {out.dtype} doesn't match expected {expected_dtype}")

    if not is_writeable_array(out):
        raise ValueError("Output array `out` is not writable. Try using cupy or pytorch for the output array.")
    if not is_contiguous(out):
        raise ValueError("Output array `out` must be contiguous.")

    assert modulation_freq_hz is not None

    # Use the unified beamform function that handles mixed CPU/GPU arrays automatically
    nb_beamform(
        channel_data=channel_data,
        rx_coords_m=rx_coords_m,
        scan_coords_m=scan_coords_m,
        tx_wave_arrivals_s=tx_wave_arrivals_s,
        out=out,
        f_number=f_number,
        rx_start_s=rx_start_s,
        sampling_freq_hz=sampling_freq_hz,
        sound_speed_m_s=sound_speed_m_s,
        modulation_freq_hz=modulation_freq_hz,
        tukey_alpha=tukey_alpha,
    )

    return cast(Array, out)
