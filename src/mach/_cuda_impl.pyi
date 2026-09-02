import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike

__nvcc_version__: str = "12.4.131"

class InterpolationType(enum.Enum):
    NearestNeighbor = 0
    """Use nearest neighbor interpolation (fastest)"""

    Linear = 1
    """Use linear interpolation (default, good balance)"""

    Quadratic = 2
    """Use quadratic interpolation (higher quality)"""

NearestNeighbor: InterpolationType = InterpolationType.NearestNeighbor

Linear: InterpolationType = InterpolationType.Linear

Quadratic: InterpolationType = InterpolationType.Quadratic

@overload
def beamform(
    channel_data: Annotated[ArrayLike, dict(dtype="complex64", shape=(None, None, None), order="C", writable=False)],
    rx_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    scan_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    tx_wave_arrivals_s: Annotated[ArrayLike, dict(dtype="float32", shape=(None), order="C", writable=False)],
    out: Annotated[ArrayLike, dict(dtype="complex64", shape=(None, None), order="C")],
    f_number: float,
    rx_start_s: float,
    sampling_freq_hz: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
    use_inverted_kernel: bool = True,
) -> None: ...
@overload
def beamform(
    channel_data: Annotated[ArrayLike, dict(dtype="float32", shape=(None, None, None), order="C", writable=False)],
    rx_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    scan_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    tx_wave_arrivals_s: Annotated[ArrayLike, dict(dtype="float32", shape=(None), order="C", writable=False)],
    out: Annotated[ArrayLike, dict(dtype="float32", shape=(None, None), order="C")],
    f_number: float,
    rx_start_s: float,
    sampling_freq_hz: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float = 0.0,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
    use_inverted_kernel: bool = True,
) -> None: ...
def beamform_fp16(
    channel_data: Annotated[ArrayLike, dict(dtype="uint16", shape=(None, None, None), order="C", writable=False)],
    rx_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    scan_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    tx_wave_arrivals_s: Annotated[ArrayLike, dict(dtype="float32", shape=(None), order="C", writable=False)],
    out: Annotated[ArrayLike, dict(dtype="complex64", shape=(None, None), order="C")],
    f_number: float,
    rx_start_s: float,
    sampling_freq_hz: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
    use_inverted_kernel: bool = True,
) -> None:
    """
    I/Q beamforming with FP16 (half2) channel-data storage (GPU arrays only).
    channel_data is the complex64 data as interleaved float16 (re, im) pairs viewed as uint16, shape (n_rx, n_samples, 2 * n_frames): iq.view(cp.float32).astype(cp.float16).view(cp.uint16). Compute and out stay float32 / complex64; out is accumulated into and must be zero-initialised by the caller.
    """

def beamform_vjp(
    channel_data: Annotated[ArrayLike, dict(dtype="complex64", shape=(None, None, None), order="C", writable=False)],
    rx_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    scan_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C", writable=False)],
    tx_wave_arrivals_s: Annotated[ArrayLike, dict(dtype="float32", shape=(None), order="C", writable=False)],
    grad_out: Annotated[ArrayLike, dict(dtype="complex64", shape=(None, None), order="C", writable=False)],
    grad_channel_data: Annotated[ArrayLike, dict(dtype="complex64", shape=(None, None, None), order="C")] | None = None,
    grad_tx_wave_arrivals_s: Annotated[ArrayLike, dict(dtype="float32", shape=(None), order="C")] | None = None,
    grad_scan_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C")] | None = None,
    grad_rx_coords_m: Annotated[ArrayLike, dict(dtype="float32", shape=(None, 3), order="C")] | None = None,
    grad_sound_speed_m_s: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")] | None = None,
    grad_rx_start_s: Annotated[ArrayLike, dict(dtype="float64", shape=(None), order="C")] | None = None,
    f_number: float,
    rx_start_s: float,
    sampling_freq_hz: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
) -> None:
    """
    Vector-Jacobian product (backward pass) of beamform() for complex64 channel data (GPU arrays only, inverted-kernel layout: nearest/linear interpolation, channel_data.shape[2] a multiple of 4 frames >= grad_out.shape[1]).

    grad_out is dL/dRe(out) + j dL/dIm(out). Each gradient buffer that is given is accumulated into and must be zero-initialised by the caller: grad_channel_data (the adjoint / backprojection), grad_tx_wave_arrivals_s, grad_scan_coords_m, grad_rx_coords_m, grad_sound_speed_m_s (float64, shape (1,)) and grad_rx_start_s (float64, shape (1,)). The aperture, sample bounds and apodization weight are treated as constants with respect to the geometry.
    """
