"""Experimental. This API is not stable and may change."""

from enum import Enum
from typing import Optional, TypeVar, cast

from array_api_compat import array_namespace
from jaxtyping import Num, Real

from mach import kernel
from mach._array_api import ArrayAPIConformant

# Type variable for array types
Array = TypeVar("Array", bound=ArrayAPIConformant)


class DLPackDevice(int, Enum):
    """Enum for the different DLPack device types.

    Port of:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L76-L80
    """

    CPU = 1
    CUDA = 2


def beamform(
    channel_data: Num[Array, "n_transmits n_rx n_samples n_frames"],
    rx_coords_m: Real[Array, "n_rx xyz=3"],
    scan_coords_m: Real[Array, "n_points xyz=3"],
    tx_wave_arrivals_s: Real[Array, "n_transmits n_points"],
    out: Optional[Num[Array, "n_points n_frames"]] = None,
    *,
    rx_start_s: Real[Array, " n_transmits"],
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: Optional[float] = None,
    tukey_alpha: float = 0.5,
) -> Num[Array, "n_points n_frames"]:
    """Wrapper around kernel.beamform that includes coherent compounding.

    The implementation takes some shortcuts for quick prototyping.

    Args:
        channel_data: like kernel.beamform channel_data, but with an extra first dimension for transmits
        tx_wave_arrivals_s: like kernel.beamform tx_wave_arrivals_s, but with an extra first dimension for transmits
    See kernel.beamform for other argument descriptions.

    Returns:
        beamformed+compounded data with shape (n_points, n_frames)
    """

    xp_data = array_namespace(channel_data)
    n_points = scan_coords_m.shape[0]
    n_transmits, _, _, n_frames = channel_data.shape
    is_complex = (channel_data.dtype == xp_data.complex64) or (channel_data.dtype == xp_data.complex128)
    output_dtype = xp_data.complex64 if is_complex else xp_data.float32

    # Allocate output array if not provided
    if out is None:
        out = xp_data.zeros((n_points, n_frames), dtype=output_dtype)
    else:
        # Validate output array dtype matches input
        expected_dtype = xp_data.complex64 if is_complex else xp_data.float32
        if out.dtype != expected_dtype:
            raise ValueError(f"Output array dtype {out.dtype} doesn't match expected {expected_dtype}")

    # TODO: clean this up
    # Because we accumulate, we need to ensure the output array is on the GPU
    # so it accumulates in place
    out_orig = out
    if out_orig.__dlpack_device__()[0] != DLPackDevice.CUDA:
        import cupy as cp

        out = cp.zeros_like(out_orig)

    # Compounding is: simply summing the beamform data into the same output array
    for transmit_idx in range(n_transmits):
        # Extract single-transmit data
        single_channel_data = channel_data[transmit_idx]

        # Call single-transmit beamform
        _ = kernel.beamform(
            single_channel_data,
            rx_coords_m,
            scan_coords_m,
            tx_wave_arrivals_s[transmit_idx],
            out=out,
            rx_start_s=rx_start_s,
            sampling_freq_hz=sampling_freq_hz,
            f_number=f_number,
            sound_speed_m_s=sound_speed_m_s,
            modulation_freq_hz=modulation_freq_hz,
            tukey_alpha=tukey_alpha,
        )

    # Move the data back to the dedicated output array
    if out_orig.__dlpack_device__()[0] != DLPackDevice.CUDA:
        out_orig[:] = out.get()
        return out_orig

    return cast(Array, out)
