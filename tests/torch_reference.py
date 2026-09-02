"""Pure-PyTorch reference delay-and-sum that mirrors mach's I/Q kernel semantics.

Used by the autograd tests as ground truth: the forward is checked against
``mach.beamform`` and PyTorch autograd of this function provides reference
gradients for the CUDA vector-Jacobian product.

Mirrors ``beamformKernelInvIQ`` / ``calculateTxRxDelayAndApodization``:

* tau = tx_wave_arrivals_s[v] + |rx[e] - scan[v]| / c
* receive aperture: horizontal distance (x, y) <= z_v / (2 f#), Tukey taper on the normalised
  horizontal distance (``tukey_apod_weight``); ``tukey_alpha == 0`` means no apodization
* sample index = (tau - rx_start_s) * fs; linear (``lerp``) or nearest (round half to even)
  interpolation with the kernel's bounds tests; out-of-bounds taps contribute nothing
* phase rotation exp(+j 2 pi f_mod tau) applied to the interpolated sample

Everything is vectorised over (voxel, element, frame), so keep problem sizes small.
"""

import math

import torch

from mach.kernel import InterpolationType


def tukey_apod_weight(r_norm: torch.Tensor, alpha: float) -> torch.Tensor:
    """Half Tukey window on r_norm in [0, 1]; zero outside (matches the CUDA helper)."""
    weight = torch.ones_like(r_norm)
    if alpha > 0:
        taper = 0.5 - 0.5 * torch.cos(math.pi * (1.0 - r_norm) / alpha)
        weight = torch.where(r_norm > (1.0 - alpha), taper, weight)
    return torch.where((r_norm < 0) | (r_norm > 1), torch.zeros_like(weight), weight)


def beamform_reference(
    channel_data: torch.Tensor,
    rx_coords_m: torch.Tensor,
    scan_coords_m: torch.Tensor,
    tx_wave_arrivals_s: torch.Tensor,
    *,
    sound_speed_m_s,
    f_number: float,
    rx_start_s: float,
    sampling_freq_hz: float,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
    rx_delays_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delay-and-sum of ``channel_data`` (n_rx, n_samples, n_frames) -> (n_scan, n_frames).

    ``sound_speed_m_s`` may be a 0-d tensor to differentiate with respect to it. As in the CUDA
    VJP, the Tukey weight, the aperture mask and the bounds masks are constants with respect to the
    geometry; ``rx_delays_s`` (n_rx,) adds a per-element receive delay.
    """
    n_rx, n_samples, _ = channel_data.shape
    real_dtype = rx_coords_m.dtype
    c = torch.as_tensor(sound_speed_m_s, dtype=real_dtype, device=channel_data.device)

    diff = rx_coords_m[None, :, :] - scan_coords_m[:, None, :]  # (V, E, 3)
    horizontal_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
    aperture_radius = scan_coords_m[:, 2] / (2.0 * f_number)  # (V,)
    in_aperture = horizontal_sq <= aperture_radius[:, None] ** 2
    distance = torch.sqrt(horizontal_sq + diff[..., 2] ** 2)
    tau = tx_wave_arrivals_s[:, None] + distance / c  # (V, E)
    if rx_delays_s is not None:
        tau = tau + rx_delays_s[None, :]  # per-element receive delay (phase screen)

    if tukey_alpha > 0:
        weight = tukey_apod_weight(torch.sqrt(horizontal_sq) / aperture_radius[:, None], tukey_alpha).detach()
        valid = in_aperture & (weight != 0)
    else:
        weight = torch.ones_like(tau)
        valid = in_aperture

    sample_idx = (tau - rx_start_s) * sampling_freq_hz
    element_idx = torch.arange(n_rx, device=channel_data.device)[None, :].expand_as(tau)
    if interp_type == InterpolationType.Linear:
        valid = valid & (sample_idx >= 0) & (sample_idx <= n_samples - 1)
        idx_floor = torch.floor(sample_idx).long().clamp(0, n_samples - 1)
        idx_ceil = torch.ceil(sample_idx).long().clamp(0, n_samples - 1)
        lerp_alpha = (sample_idx - idx_floor.to(sample_idx.dtype))[..., None]
        lo = channel_data[element_idx, idx_floor]  # (V, E, F)
        hi = channel_data[element_idx, idx_ceil]
        sample = lo + lerp_alpha * (hi - lo)
    elif interp_type == InterpolationType.NearestNeighbor:
        valid = valid & (sample_idx >= -0.5) & (sample_idx <= n_samples - 0.5)
        idx_round = torch.round(sample_idx).long().clamp(0, n_samples - 1)
        sample = channel_data[element_idx, idx_round]
    else:
        raise NotImplementedError("reference covers nearest and linear interpolation only")

    complex_weight = weight.to(sample.dtype) * valid.to(sample.dtype)
    if modulation_freq_hz != 0.0:
        phase = 2.0 * math.pi * modulation_freq_hz * tau
        complex_weight = complex_weight * torch.polar(torch.ones_like(phase), phase).to(sample.dtype)
    return (complex_weight[..., None] * sample).sum(dim=1)


def simulate_point_scatterers_iq(
    rx_coords_m: torch.Tensor,
    scatterers_m: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    sound_speed_m_s: float,
    f0_hz: float,
    sampling_freq_hz: float,
    n_samples: int,
    n_frames: int = 1,
    pulse_sigma_s: float,
) -> torch.Tensor:
    """Baseband I/Q channel data from point scatterers insonified by a plane wave along +z.

    d[e, n, f] = sum_s a_s exp(-((t_n - tau_se) / sigma)^2 / 2) exp(-j 2 pi f0 tau_se),
    with tau_se = z_s / c + |rx_e - x_s| / c and t_n = n / fs (rx_start_s = 0). Beamforming with
    ``modulation_freq_hz=f0`` and ``tx_wave_arrivals_s = z / c`` undoes the carrier phase, so the
    delay-and-sum is coherent exactly at the true sound speed: a focusing problem whose image energy
    depends strongly and smoothly on the delay parameters.
    """
    device = rx_coords_m.device
    t = torch.arange(n_samples, device=device, dtype=torch.float64) / sampling_freq_hz
    diff = rx_coords_m[:, None, :].double() - scatterers_m[None, :, :].double()  # (E, S, 3)
    tau = (scatterers_m[None, :, 2].double() + torch.linalg.norm(diff, dim=-1)) / sound_speed_m_s  # (E, S)
    envelope = torch.exp(-0.5 * ((t[None, :, None] - tau[:, None, :]) / pulse_sigma_s) ** 2)  # (E, T, S)
    carrier = torch.exp(-1j * 2 * math.pi * f0_hz * tau)[:, None, :]
    data = (envelope * carrier * amplitudes.double()[None, None, :]).sum(-1)  # (E, T)
    return data[:, :, None].expand(-1, -1, n_frames).to(torch.complex64).contiguous()
