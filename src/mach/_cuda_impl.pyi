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
) -> None: ...
