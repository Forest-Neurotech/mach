"""Benchmark the backward pass (beamform_vjp) against the forward at realistic sizes.

    python tests/bench_autograd.py [--small]

Times the inverted-loop forward and the three VJP variants (channel-data adjoint only, delay
gradients only, both) on random I/Q data, median of 3 runs each.
"""

import argparse
import statistics

import torch

from mach._cuda_impl import beamform as nb_beamform
from mach._cuda_impl import beamform_vjp

CONFIGS = [
    # label, n_rx, n_samples, n_frames, n_voxels
    ("128 ch, 1024 samples, 4 frames, 120k voxels (imaging / autofocus shape)", 128, 1024, 4, 120_000),
    ("256 ch, 512 samples, 64 frames, 40k voxels", 256, 512, 64, 40_000),
    ("1024 ch, 256 samples, 128 frames, 128^3 voxels", 1024, 256, 128, 128**3),
]
KW = {
    "f_number": 1.0,
    "rx_start_s": 0.0,
    "sampling_freq_hz": 7.8e6,
    "sound_speed_m_s": 1540.0,
    "modulation_freq_hz": 7.8e6,
    "tukey_alpha": 0.5,
}


def timeit(fn, repeats=3):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def run(label, n_rx, n_samples, n_frames, n_voxels):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    chan = torch.complex(
        torch.randn((n_rx, n_samples, n_frames), generator=g, device=dev),
        torch.randn((n_rx, n_samples, n_frames), generator=g, device=dev),
    )
    rx = torch.rand((n_rx, 3), generator=g, device=dev) * 1e-2 - 5e-3
    rx[:, 2] = 0.0
    scan = torch.rand((n_voxels, 3), generator=g, device=dev) * torch.tensor(
        [1e-2, 1e-2, 1e-2], device=dev
    ) + torch.tensor([-5e-3, -5e-3, 8e-3], device=dev)
    tx = (scan[:, 2] / KW["sound_speed_m_s"]).float()
    out = torch.zeros((n_voxels, n_frames), dtype=torch.complex64, device=dev)
    grad_out = torch.complex(
        torch.randn(out.shape, generator=g, device=dev), torch.randn(out.shape, generator=g, device=dev)
    )
    grad_chan = torch.zeros_like(chan)
    grad_tx = torch.zeros_like(tx)
    grad_scan = torch.zeros_like(scan)
    grad_rx = torch.zeros_like(rx)
    grad_c = torch.zeros(1, dtype=torch.float64, device=dev)

    def forward():
        out.zero_()
        nb_beamform(chan, rx, scan, tx, out, **KW)

    def vjp(data, tau):
        return lambda: beamform_vjp(
            chan,
            rx,
            scan,
            tx,
            grad_out,
            grad_channel_data=grad_chan if data else None,
            grad_tx_wave_arrivals_s=grad_tx if tau else None,
            grad_scan_coords_m=grad_scan if tau else None,
            grad_rx_coords_m=grad_rx if tau else None,
            grad_sound_speed_m_s=grad_c if tau else None,
            **KW,
        )

    t_fwd = timeit(forward)
    t_data = timeit(vjp(True, False))
    t_tau = timeit(vjp(False, True))
    t_both = timeit(vjp(True, True))
    print(f"\n{label}")
    print(f"  forward (inverted kernel)      {t_fwd:9.1f} ms")
    print(f"  vjp: channel-data adjoint only {t_data:9.1f} ms  ({t_data / t_fwd:4.1f}x forward)")
    print(f"  vjp: delay gradients only      {t_tau:9.1f} ms  ({t_tau / t_fwd:4.1f}x forward)")
    print(f"  vjp: both                      {t_both:9.1f} ms  ({t_both / t_fwd:4.1f}x forward)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="skip the 128^3-voxel configuration")
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, {props.total_memory / 2**30:.0f} GiB")
    for cfg in CONFIGS[: -1 if args.small else None]:
        run(*cfg)
