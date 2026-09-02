"""PyTorch autograd support for the I/Q beamformer (experimental).

:func:`beamform` wraps the CUDA forward kernel and its hand-written vector-Jacobian
product so that gradients flow from the beamformed image back to

* ``channel_data`` (the adjoint of delay-and-sum, i.e. backprojection),
* ``tx_wave_arrivals_s``, ``scan_coords_m`` and ``rx_coords_m`` (through the delays),
* ``sound_speed_m_s`` and ``rx_start_s`` when they are passed as 0-d tensors,
* ``rx_delays_s``, an optional per-element receive delay added to every arrival time
  (the phase-screen aberration model), whose gradient is the per-element sum of dL/dtau.

The receive aperture, the sample-bounds masks and the Tukey apodization weight are
treated as constants with respect to the geometry (the standard convention for
differentiable delay-and-sum). Nearest-neighbour interpolation has no interpolant
slope, so its delay gradients only carry the phase-rotation term.

Requirements (the inverted-loop kernel layout): CUDA tensors, complex64 channel data,
nearest or linear interpolation, and ``channel_data.shape[2]`` (the frame stride) a
multiple of :data:`FRAMES_PER_CHUNK` that is at least the number of frames beamformed.
Allocate the acquisition buffer with a padded last axis if the frame count is not a
multiple of four; the padding lanes are ignored and receive zero gradient.

The public function name and arguments mirror :func:`mach.kernel.beamform`.
"""

from __future__ import annotations

import math

import torch

from mach import _cuda_impl
from mach.kernel import InterpolationType

__all__ = ["FRAMES_PER_CHUNK", "beamform"]

FRAMES_PER_CHUNK = 4
"""Frames per 128-bit chunk in the inverted kernel: the frame stride must be a multiple of this."""


def _as_float32(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor (mach.autograd is GPU-only)")
    return x.to(torch.float32).contiguous()


def check_layout(channel_data: torch.Tensor, n_frames: int, interp_type: InterpolationType) -> None:
    """Raise if ``channel_data`` cannot take the inverted-kernel (and hence the backward) path."""
    if channel_data.dtype != torch.complex64:
        raise TypeError(f"channel_data must be complex64 (I/Q), got {channel_data.dtype}")
    if not channel_data.is_cuda:
        raise ValueError("channel_data must be a CUDA tensor (mach.autograd is GPU-only)")
    if channel_data.ndim != 3:
        raise ValueError(
            f"channel_data must have shape (n_rx, n_samples, frame_stride), got {tuple(channel_data.shape)}"
        )
    if interp_type == InterpolationType.Quadratic:
        raise ValueError("quadratic interpolation has no backward kernel; use nearest or linear")
    frame_stride = channel_data.shape[2]
    if n_frames <= 0 or n_frames > frame_stride:
        raise ValueError(f"n_frames must be in [1, channel_data.shape[2]={frame_stride}], got {n_frames}")
    if frame_stride % FRAMES_PER_CHUNK != 0:
        raise ValueError(
            f"channel_data.shape[2]={frame_stride} must be a multiple of {FRAMES_PER_CHUNK} frames "
            "(allocate the buffer with a padded last axis; padding lanes are ignored)"
        )
    if channel_data.data_ptr() % 16 != 0:
        raise ValueError("channel_data must be 16-byte aligned (not an offset view)")


class _BeamformIQ(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        channel_data,
        rx_coords_m,
        scan_coords_m,
        tx_wave_arrivals_s,
        rx_delays_s,
        sound_speed_m_s,
        rx_start_s,
        n_frames,
        f_number,
        sampling_freq_hz,
        modulation_freq_hz,
        tukey_alpha,
        interp_type,
    ):
        c = float(sound_speed_m_s)
        t0 = float(rx_start_s)
        out = torch.zeros((scan_coords_m.shape[0], n_frames), dtype=torch.complex64, device=channel_data.device)
        _cuda_impl.beamform(
            channel_data,
            rx_coords_m,
            scan_coords_m,
            tx_wave_arrivals_s,
            out,
            f_number=f_number,
            rx_start_s=t0,
            sampling_freq_hz=sampling_freq_hz,
            sound_speed_m_s=c,
            modulation_freq_hz=modulation_freq_hz,
            tukey_alpha=tukey_alpha,
            interp_type=interp_type,
            rx_delays_s=rx_delays_s,
        )
        ctx.save_for_backward(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, rx_delays_s)
        ctx.scalars = (c, t0, f_number, sampling_freq_hz, modulation_freq_hz, tukey_alpha, interp_type)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, rx_delays_s = ctx.saved_tensors
        c, t0, f_number, sampling_freq_hz, modulation_freq_hz, tukey_alpha, interp_type = ctx.scalars
        need = ctx.needs_input_grad
        device = channel_data.device
        grad_out = grad_out.to(torch.complex64).contiguous()

        grad_channel = torch.zeros_like(channel_data) if need[0] else None
        grad_rx = torch.zeros_like(rx_coords_m) if need[1] else None
        grad_scan = torch.zeros_like(scan_coords_m) if need[2] else None
        grad_tx = torch.zeros_like(tx_wave_arrivals_s) if need[3] else None
        grad_delays = torch.zeros_like(rx_delays_s) if (need[4] and rx_delays_s is not None) else None
        grad_c = torch.zeros(1, dtype=torch.float64, device=device) if need[5] else None
        grad_t0 = torch.zeros(1, dtype=torch.float64, device=device) if need[6] else None

        if any(g is not None for g in (grad_channel, grad_rx, grad_scan, grad_tx, grad_delays, grad_c, grad_t0)):
            _cuda_impl.beamform_vjp(
                channel_data,
                rx_coords_m,
                scan_coords_m,
                tx_wave_arrivals_s,
                grad_out,
                grad_channel_data=grad_channel,
                grad_tx_wave_arrivals_s=grad_tx,
                grad_scan_coords_m=grad_scan,
                grad_rx_coords_m=grad_rx,
                grad_sound_speed_m_s=grad_c,
                grad_rx_start_s=grad_t0,
                rx_delays_s=rx_delays_s,
                grad_rx_delays_s=grad_delays,
                f_number=f_number,
                rx_start_s=t0,
                sampling_freq_hz=sampling_freq_hz,
                sound_speed_m_s=c,
                modulation_freq_hz=modulation_freq_hz,
                tukey_alpha=tukey_alpha,
                interp_type=interp_type,
            )

        def scalar(g):
            return None if g is None else g[0]

        return (
            grad_channel,
            grad_rx,
            grad_scan,
            grad_tx,
            grad_delays,
            scalar(grad_c),
            scalar(grad_t0),
            None,  # n_frames
            None,  # f_number
            None,  # sampling_freq_hz
            None,  # modulation_freq_hz
            None,  # tukey_alpha
            None,  # interp_type
        )


def beamform(
    channel_data: torch.Tensor,
    rx_coords_m: torch.Tensor,
    scan_coords_m: torch.Tensor,
    tx_wave_arrivals_s: torch.Tensor,
    *,
    rx_start_s: float | torch.Tensor,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float | torch.Tensor,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
    interp_type: InterpolationType = InterpolationType.Linear,
    rx_delays_s: torch.Tensor | None = None,
    n_frames: int | None = None,
) -> torch.Tensor:
    """Differentiable delay-and-sum of I/Q ``channel_data`` -> ``(n_scan, n_frames)`` complex64.

    Same semantics as :func:`mach.kernel.beamform` for complex data. ``sound_speed_m_s`` and
    ``rx_start_s`` may be 0-d tensors with ``requires_grad`` to obtain their gradients.
    ``rx_delays_s`` (n_rx,) adds a per-element receive delay to every arrival time. ``n_frames``
    defaults to ``channel_data.shape[2]`` and may be smaller when the frame axis is padded (see
    the module docstring).
    """
    if n_frames is None:
        n_frames = int(channel_data.shape[2])
    if isinstance(channel_data, torch.Tensor) and channel_data.dtype == torch.complex128:
        channel_data = channel_data.to(torch.complex64)
    check_layout(channel_data, n_frames, interp_type)
    channel_data = channel_data.contiguous()
    rx_coords_m = _as_float32(rx_coords_m, "rx_coords_m")
    scan_coords_m = _as_float32(scan_coords_m, "scan_coords_m")
    tx_wave_arrivals_s = _as_float32(tx_wave_arrivals_s, "tx_wave_arrivals_s")
    if rx_delays_s is not None:
        rx_delays_s = _as_float32(rx_delays_s, "rx_delays_s")
        if rx_delays_s.shape != (rx_coords_m.shape[0],):
            raise ValueError(
                f"rx_delays_s must have shape (n_rx,) = ({rx_coords_m.shape[0]},), got {tuple(rx_delays_s.shape)}"
            )
    if modulation_freq_hz is None:
        raise ValueError("modulation_freq_hz is required for I/Q data; set it to 0 if no demodulation was used")
    if not math.isfinite(float(sound_speed_m_s)) or float(sound_speed_m_s) <= 0:
        raise ValueError("sound_speed_m_s must be a positive finite number")
    return _BeamformIQ.apply(
        channel_data,
        rx_coords_m,
        scan_coords_m,
        tx_wave_arrivals_s,
        rx_delays_s,
        sound_speed_m_s,
        rx_start_s,
        n_frames,
        float(f_number),
        float(sampling_freq_hz),
        float(modulation_freq_hz),
        float(tukey_alpha),
        interp_type,
    )
