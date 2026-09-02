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
nearest or linear interpolation, and a frame stride ``channel_data.shape[2]`` that keeps
every sample row 16-byte aligned and leaves room for a whole :data:`FRAMES_PER_CHUNK`
chunk past the frames beamformed. :func:`pad_frames` produces such a buffer; the padding
lanes are ignored and receive zero gradient.

:func:`sharpness` is the image objective used by the autofocus examples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mach import _cuda_impl, kernel
from mach.kernel import InterpolationType

__all__ = ["FRAMES_PER_CHUNK", "beamform", "check_layout", "pad_frames", "sharpness"]

FRAMES_PER_CHUNK = 4
"""Frames per 128-bit chunk in the inverted kernel."""


def pad_frames(channel_data: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Zero-pad the frame axis to a multiple of :data:`FRAMES_PER_CHUNK`; returns (buffer, n_frames)."""
    n_frames = int(channel_data.shape[-1])
    pad = (-n_frames) % FRAMES_PER_CHUNK
    if pad:
        channel_data = torch.cat([channel_data, channel_data.new_zeros((*channel_data.shape[:-1], pad))], dim=-1)
    return channel_data.contiguous(), n_frames


def sharpness(image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Normalised sharpness sum|I|^4 / (sum|I|^2)^2 of an image: scale-free, maximal when the energy is
    concentrated. Unlike the image energy it is not invariant to phase errors, so it works as an
    autofocus objective for plane-wave images. ``mask`` restricts it to a boolean subset of voxels."""
    intensity = image.abs().pow(2)
    if mask is not None:
        intensity = intensity[mask]
    return intensity.pow(2).sum() / intensity.sum().pow(2)


def check_layout(channel_data: torch.Tensor, n_frames: int, interp_type: InterpolationType) -> None:
    """Raise if ``channel_data`` cannot take the inverted-kernel (and hence the backward) path.

    Mirrors the kernel's own layout rule (``inverted_layout_reason`` in kernel.cu)."""
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
    frame_stride = int(channel_data.shape[2])
    if not 0 < n_frames <= frame_stride:
        raise ValueError(f"n_frames must be in [1, channel_data.shape[2]={frame_stride}], got {n_frames}")
    chunked = -(-n_frames // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK
    if frame_stride % 2 != 0 or chunked > frame_stride:
        raise ValueError(
            f"channel_data.shape[2]={frame_stride} must be even and at least {chunked} for n_frames={n_frames}: "
            f"pad the frame axis to a multiple of {FRAMES_PER_CHUNK} (see pad_frames)"
        )
    if channel_data.data_ptr() % 16 != 0:
        raise ValueError("channel_data must be 16-byte aligned (not an offset view)")


@dataclass(frozen=True)
class _Params:
    """Non-differentiable arguments, converted once (a 0-d CUDA tensor costs a device sync per float())."""

    n_frames: int
    f_number: float
    rx_start_s: float
    sampling_freq_hz: float
    sound_speed_m_s: float
    modulation_freq_hz: float
    tukey_alpha: float
    interp_type: InterpolationType


# Differentiable inputs, in the order _BeamformIQ.apply receives them (the params object comes last).
_GRAD_INPUTS = (
    "channel_data",
    "rx_coords_m",
    "scan_coords_m",
    "tx_wave_arrivals_s",
    "rx_delays_s",
    "sound_speed_m_s",
    "rx_start_s",
)


class _BeamformIQ(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        channel_data,
        rx_coords_m,
        scan_coords_m,
        tx_wave_arrivals_s,
        rx_delays_s,
        sound_speed_m_s,
        rx_start_s,
        params,
    ):  # type: ignore[override]
        out = torch.zeros((scan_coords_m.shape[0], params.n_frames), dtype=torch.complex64, device=channel_data.device)
        # Launch on torch's current stream without a device-wide synchronize: torch orders the surrounding
        # work on that stream, so the host keeps queueing while the kernel runs.
        kernel.beamform(
            channel_data,
            rx_coords_m,
            scan_coords_m,
            tx_wave_arrivals_s,
            out,
            rx_start_s=params.rx_start_s,
            sampling_freq_hz=params.sampling_freq_hz,
            f_number=params.f_number,
            sound_speed_m_s=params.sound_speed_m_s,
            modulation_freq_hz=params.modulation_freq_hz,
            tukey_alpha=params.tukey_alpha,
            interp_type=params.interp_type,
            rx_delays_s=rx_delays_s,
            stream=torch.cuda.current_stream(channel_data.device).cuda_stream,
            synchronize=False,
        )
        ctx.save_for_backward(channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, rx_delays_s)
        ctx.params = params
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, rx_delays_s = ctx.saved_tensors
        p = ctx.params
        need = dict(zip(_GRAD_INPUTS, ctx.needs_input_grad, strict=False))
        device = channel_data.device

        def like(name, template):
            return torch.zeros_like(template) if (need[name] and template is not None) else None

        def scalar(name):
            return torch.zeros(1, dtype=torch.float64, device=device) if need[name] else None

        grads = {
            "channel_data": like("channel_data", channel_data),
            "rx_coords_m": like("rx_coords_m", rx_coords_m),
            "scan_coords_m": like("scan_coords_m", scan_coords_m),
            "tx_wave_arrivals_s": like("tx_wave_arrivals_s", tx_wave_arrivals_s),
            "rx_delays_s": like("rx_delays_s", rx_delays_s),
            "sound_speed_m_s": scalar("sound_speed_m_s"),
            "rx_start_s": scalar("rx_start_s"),
        }
        if any(g is not None for g in grads.values()):
            _cuda_impl.beamform_vjp(
                channel_data,
                rx_coords_m,
                scan_coords_m,
                tx_wave_arrivals_s,
                grad_out.to(torch.complex64).contiguous(),
                grad_channel_data=grads["channel_data"],
                grad_tx_wave_arrivals_s=grads["tx_wave_arrivals_s"],
                grad_scan_coords_m=grads["scan_coords_m"],
                grad_rx_coords_m=grads["rx_coords_m"],
                grad_sound_speed_m_s=grads["sound_speed_m_s"],
                grad_rx_start_s=grads["rx_start_s"],
                rx_delays_s=rx_delays_s,
                grad_rx_delays_s=grads["rx_delays_s"],
                f_number=p.f_number,
                rx_start_s=p.rx_start_s,
                sampling_freq_hz=p.sampling_freq_hz,
                sound_speed_m_s=p.sound_speed_m_s,
                modulation_freq_hz=p.modulation_freq_hz,
                tukey_alpha=p.tukey_alpha,
                interp_type=p.interp_type,
                stream=torch.cuda.current_stream(device).cuda_stream,
                synchronize=False,
            )
        for name in ("sound_speed_m_s", "rx_start_s"):
            if grads[name] is not None:
                grads[name] = grads[name][0]  # 0-d, matching a 0-d input tensor
        return (*(grads[name] for name in _GRAD_INPUTS), None)


def _as_float32(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor (mach.autograd is GPU-only)")
    return x.to(torch.float32).contiguous()


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
    :func:`pad_frames`).
    """
    if n_frames is None:
        n_frames = int(channel_data.shape[2])
    if channel_data.dtype == torch.complex128:
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
    c = float(sound_speed_m_s)
    if not math.isfinite(c) or c <= 0:
        raise ValueError("sound_speed_m_s must be a positive finite number")
    params = _Params(
        n_frames=n_frames,
        f_number=float(f_number),
        rx_start_s=float(rx_start_s),
        sampling_freq_hz=float(sampling_freq_hz),
        sound_speed_m_s=c,
        modulation_freq_hz=float(modulation_freq_hz),
        tukey_alpha=float(tukey_alpha),
        interp_type=interp_type,
    )
    return _BeamformIQ.apply(
        channel_data, rx_coords_m, scan_coords_m, tx_wave_arrivals_s, rx_delays_s, sound_speed_m_s, rx_start_s, params
    )
