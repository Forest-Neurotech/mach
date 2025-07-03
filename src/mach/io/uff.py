"""Data loader for UFF (ultrasound file format) files.


Usage: pass in a scan-object from:
https://github.com/magnusdk/pyuff_ustb
"""

from typing import Any, Optional, TypeVar

import einops
import numpy as np
from array_api_compat import array_namespace
from scipy.signal import hilbert

from mach._array_api import ArrayAPIConformant
from mach.geometry import ultrasound_angles_to_cartesian
from mach.wavefront import plane

Array = TypeVar("Array", bound=ArrayAPIConformant)


def extract_wave_directions(sequence: list[Any], xp) -> list:
    """
    Extract wave propagation directions from ultrasound sequence.

    Args:
        sequence: List of wave objects containing source information
        xp: Array namespace (numpy, cupy, etc.)

    Returns:
        List of direction vectors as arrays
    """
    azimuth = xp.asarray([wave.source.azimuth for wave in sequence])
    elevation = xp.asarray([wave.source.elevation for wave in sequence])
    return ultrasound_angles_to_cartesian(azimuth, elevation)


def compute_tx_wave_arrivals_s(
    directions: list, scan_coords_m: Array, speed_of_sound: float, origin: Optional[Array] = None, xp=None
) -> Array:
    """
    Compute transmit arrival times for plane wave imaging.

    Args:
        directions: List of wave direction vectors
        scan_coords_m: Output positions for beamforming (N, 3) in meters
        speed_of_sound: Speed of sound in the medium
        origin: Wave origin point, defaults to [0, 0, 0]
            Can also be parsed from ultrasound_angles_to_cartesian(channel_data.sequence[idx].origin)
        xp: Array namespace (optional, will be inferred from scan_coords_m)

    Returns:
        Array of transmit arrival times with shape (n_transmits, n_points)
    """
    if xp is None:
        xp = array_namespace(scan_coords_m)

    if origin is None:
        origin = xp.asarray([0.0, 0.0, 0.0])

    tx_wave_arrivals_s = []
    for direction in directions:
        # The plane function is array-API compatible, so we can pass arrays directly
        arrival = plane(origin, scan_coords_m, direction)
        tx_wave_arrivals_s.append(arrival / speed_of_sound)

    # Stack all arrivals into a single array with shape (n_transmits, n_points)
    return xp.stack(tx_wave_arrivals_s, axis=0)


def preprocess_signal(signal_data: Array, modulation_frequency: float, xp=None) -> Array:
    """
    Preprocess ultrasound signal data for multi-transmit beamforming.

    Args:
        signal_data: Raw signal data with shape (n_samples, n_elements, n_waves) or
                     (n_samples, n_elements, n_waves, n_frames)
        modulation_frequency: Modulation frequency in Hz
        xp: Array namespace (optional, will be inferred from signal_data)

    Returns:
        Preprocessed signal array with shape (n_transmits, n_receive_elements, n_samples, n_frames)
    """
    if xp is None:
        xp = array_namespace(signal_data)

    # Convert to numpy for processing operations
    signal = signal_data.get() if hasattr(signal_data, "get") else np.asarray(signal_data)

    # Apply Hilbert transform if modulation frequency is 0
    if modulation_frequency == 0:
        signal = np.array(hilbert(signal, axis=0), dtype="complex64")

    # Handle different input shapes
    if signal.ndim == 3:
        # Shape: (n_samples, n_elements, n_waves) -> add frames dimension
        signal = signal[..., np.newaxis]  # (n_samples, n_elements, n_waves, 1)
    elif signal.ndim == 4:
        # Shape: (n_samples, n_elements, n_waves, n_frames) - already correct
        pass
    else:
        raise ValueError(f"Expected signal data to have 3 or 4 dimensions, got {signal.ndim}")

    # Rearrange from (n_samples, n_elements, n_waves, n_frames) to
    # (n_transmits, n_receive_elements, n_samples, n_frames)
    signal = einops.rearrange(signal, "n_samples n_elements n_waves n_frames -> n_waves n_elements n_samples n_frames")

    # Convert back to desired array type
    return xp.asarray(signal)


def extract_sequence_delays(sequence: list[Any], xp=None) -> Array:
    """Extract delay times from ultrasound sequence.

    Args:
        sequence: List of wave objects
        xp: Array namespace (if None, will use numpy)

    Returns:
        Array of delay times with shape (n_transmits,)
    """
    delays = np.array([wave.delay for wave in sequence])

    if xp is not None:
        delays = xp.asarray(delays)

    return delays


def create_beamforming_setup(channel_data, scan, f_number: float = 1.7, xp=None) -> dict[str, Any]:
    """Create complete beamforming setup from channel data and scan parameters for all transmits.

    Args:
        channel_data: Channel data object containing signal
        scan: Scan object containing spatial parameters
        f_number: F-number for beamforming
        xp: Array namespace (if None, will use numpy)

    Returns:
        Dictionary containing all beamforming parameters for multi-transmit beamforming
    """
    # Extract basic parameters
    speed_of_sound = float(channel_data.sound_speed)
    modulation_frequency = channel_data.modulation_frequency
    sampling_freq_hz = channel_data.sampling_frequency

    # Process signal data for all transmits
    signal = preprocess_signal(channel_data.data, modulation_frequency, xp)

    # Create spatial grids
    assert scan.xyz.shape[0] == 1, f"Expected scan.xyz to be one spatial grid, but got {scan.xyz.shape}"
    scan_coords_m = scan.xyz[0]
    rx_coords_m = channel_data.probe.xyz
    if xp is not None:
        scan_coords_m = xp.asarray(scan_coords_m)
        rx_coords_m = xp.asarray(rx_coords_m)

    # Compute transmit arrivals for all transmits
    directions = extract_wave_directions(channel_data.sequence, xp or np)
    tx_wave_arrivals_s = compute_tx_wave_arrivals_s(directions, scan_coords_m, speed_of_sound, xp=xp)
    # further delay each transmit by the delay of the wave
    tx_wave_arrivals_s = tx_wave_arrivals_s + extract_sequence_delays(channel_data.sequence, xp)[:, None]

    # Account for initial_time offset (this is how vbeam handles it)
    # The initial_time represents when the first sample was acquired relative to t=0
    rx_start_s = float(channel_data.initial_time)

    return {
        "channel_data": signal,
        "rx_coords_m": rx_coords_m,
        "scan_coords_m": scan_coords_m,
        "tx_wave_arrivals_s": tx_wave_arrivals_s,
        "out": None,
        "f_number": f_number,
        "sampling_freq_hz": sampling_freq_hz,
        "sound_speed_m_s": speed_of_sound,
        "modulation_freq_hz": modulation_frequency,
        "rx_start_s": rx_start_s,
    }


def create_single_transmit_beamforming_setup(
    channel_data, scan, wave_index: int = 0, f_number: float = 1.7, xp=None
) -> dict[str, Any]:
    """
    Create beamforming setup for a single transmit (backwards compatibility).

    Args:
        channel_data: Channel data object containing signal
        scan: Scan object containing spatial parameters
        wave_index: Index of wave to use for beamforming
        f_number: F-number for beamforming
        xp: Array namespace (if None, will use numpy)

    Returns:
        Dictionary containing beamforming parameters for single transmit
    """
    # Get multi-transmit setup
    multi_setup = create_beamforming_setup(channel_data, scan, f_number, xp)

    # Extract single transmit data
    single_setup = multi_setup.copy()
    single_setup["channel_data"] = multi_setup["channel_data"][wave_index]
    single_setup["tx_wave_arrivals_s"] = multi_setup["tx_wave_arrivals_s"][wave_index]

    return single_setup
