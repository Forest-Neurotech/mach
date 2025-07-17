"""Type stubs for the nanobind-generated _cuda_impl module."""

from typing import Optional

from mach._array_api import Array

# Version information exposed by nanobind
__nvcc_version__: str

# Beamforming function from nanobind
def beamform(
    channel_data: Array,
    rx_coords_m: Array,
    scan_coords_m: Array,
    tx_wave_arrivals_s: Array,
    out: Optional[Array],
    f_number: float,
    rx_start_s: float,
    sampling_freq_hz: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
) -> Array: ...
