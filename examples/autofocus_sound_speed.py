"""Sound-speed autofocus with differentiable beamforming (``mach.autograd``).

Simulates point scatterers in a medium with c = 1540 m/s, beamforms with a wrong sound speed
and recovers the true value by gradient ascent on the image sharpness. The gradient flows
through the CUDA kernel's backward pass (receive delays) and through PyTorch for the
plane-wave transmit arrivals computed with :func:`mach.wavefront.plane`.

Requires a CUDA build of PyTorch::

    python examples/autofocus_sound_speed.py          # prints the optimisation trajectory
    python examples/autofocus_sound_speed.py --plot   # also writes autofocus_sound_speed.png
"""

import argparse
import math

import torch

from mach import wavefront
from mach.autograd import beamform, sharpness

C_TRUE_M_S = 1540.0
F0_HZ = 5e6
FS_HZ = 20e6
PULSE_SIGMA_S = 0.5e-6
N_RX, PITCH_M = 128, 0.3e-3
N_SAMPLES, N_FRAMES = 1100, 4
DEV = "cuda"


def simulate_channel_data(rx_coords_m, scatterers_m, generator):
    """Baseband I/Q echoes of point scatterers under a plane wave along +z (Gaussian pulse)."""
    t = torch.arange(N_SAMPLES, device=DEV, dtype=torch.float64) / FS_HZ
    diff = rx_coords_m[:, None, :].double() - scatterers_m[None, :, :].double()
    tau = (scatterers_m[None, :, 2].double() + torch.linalg.norm(diff, dim=-1)) / C_TRUE_M_S  # (rx, scatterer)
    envelope = torch.exp(-0.5 * ((t[None, :, None] - tau[:, None, :]) / PULSE_SIGMA_S) ** 2)
    carrier = torch.exp(-1j * 2 * math.pi * F0_HZ * tau)[:, None, :]
    data = (envelope * carrier).sum(-1)[:, :, None].expand(-1, -1, N_FRAMES)
    noise = torch.randn(data.shape, generator=generator, device=DEV, dtype=torch.float64)
    noise = torch.complex(noise, torch.randn(data.shape, generator=generator, device=DEV, dtype=torch.float64))
    return (data + 0.05 * noise).to(torch.complex64).contiguous()


def image(channel_data, rx_coords_m, scan_coords_m, plane_distance_m, sound_speed_m_s):
    """Beamform with plane-wave transmit arrivals recomputed from the trial sound speed."""
    return beamform(
        channel_data,
        rx_coords_m,
        scan_coords_m,
        (plane_distance_m / sound_speed_m_s).to(torch.float32),
        rx_start_s=0.0,
        sampling_freq_hz=FS_HZ,
        f_number=1.0,
        sound_speed_m_s=sound_speed_m_s,
        modulation_freq_hz=F0_HZ,
        tukey_alpha=0.5,
    )


def to_db(envelope):
    return 20 * torch.log10(envelope / envelope.max() + 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", action="store_true", help="write autofocus_sound_speed.png (needs matplotlib)")
    parser.add_argument("--start", type=float, default=C_TRUE_M_S + 60.0, help="initial sound speed guess (m/s)")
    args = parser.parse_args()

    generator = torch.Generator(device=DEV).manual_seed(0)
    rx = torch.zeros((N_RX, 3), device=DEV)
    rx[:, 0] = (torch.arange(N_RX, device=DEV) - (N_RX - 1) / 2) * PITCH_M
    scatterers = torch.rand((25, 3), generator=generator, device=DEV) * torch.tensor([16e-3, 0.0, 25e-3], device=DEV)
    scatterers = scatterers + torch.tensor([-8e-3, 0.0, 8e-3], device=DEV)
    channel_data = simulate_channel_data(rx, scatterers, generator)

    x = torch.linspace(-10e-3, 10e-3, 201, device=DEV)  # 0.1 mm: resolves the PSF (lambda = 0.3 mm)
    z = torch.linspace(5e-3, 35e-3, 601, device=DEV)  # 0.05 mm
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    scan = torch.stack([xx.flatten(), torch.zeros_like(xx.flatten()), zz.flatten()], dim=1)
    # transmit geometry does not depend on the trial sound speed: compute it once
    plane_distance_m = wavefront.plane(torch.zeros(3, device=DEV), scan, torch.tensor([0.0, 0.0, 1.0], device=DEV))
    print(f"{N_RX} elements, {scatterers.shape[0]} scatterers, {scan.shape[0]} voxels, {N_FRAMES} frames")

    c = torch.tensor(args.start, dtype=torch.float64, device=DEV, requires_grad=True)
    optimizer = torch.optim.Adam([c], lr=4.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    trajectory = []
    for iteration in range(100):
        optimizer.zero_grad()
        loss = -sharpness(image(channel_data, rx, scan, plane_distance_m, c))
        loss.backward()
        trajectory.append((float(c), -float(loss), float(c.grad)))
        if iteration % 10 == 0:
            print(
                f"iteration {iteration:3d}: c = {float(c):8.2f} m/s   sharpness = {-float(loss):.4e}   dS/dc = {float(c.grad):+.2e}"
            )
        optimizer.step()
        scheduler.step()
    print(f"recovered c = {float(c):.2f} m/s (true {C_TRUE_M_S:.1f}, started at {args.start:.1f})")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with torch.no_grad():
            speeds = torch.arange(1440.0, 1641.0, 5.0)
            curve = [float(sharpness(image(channel_data, rx, scan, plane_distance_m, float(s)))) for s in speeds]
            before = (
                image(channel_data, rx, scan, plane_distance_m, args.start).abs()[:, 0].reshape(len(x), len(z)).T.cpu()
            )
            after = (
                image(channel_data, rx, scan, plane_distance_m, float(c)).abs()[:, 0].reshape(len(x), len(z)).T.cpu()
            )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].plot(speeds, curve, label="sharpness")
        axes[0].plot([p[0] for p in trajectory], [p[1] for p in trajectory], ".-", label="gradient ascent")
        axes[0].axvline(C_TRUE_M_S, color="k", ls="--", lw=0.8)
        axes[0].set_xlabel("sound speed (m/s)")
        axes[0].legend()
        extent = [x[0].item() * 1e3, x[-1].item() * 1e3, z[-1].item() * 1e3, z[0].item() * 1e3]
        for ax, img, title in (
            (axes[1], before, f"c = {args.start:.0f} m/s"),
            (axes[2], after, f"c = {float(c):.1f} m/s"),
        ):
            ax.imshow(to_db(img), extent=extent, cmap="gray", vmin=-50, vmax=0, aspect="equal")
            ax.set_title(title)
            ax.set_xlabel("x (mm)")
        axes[1].set_ylabel("z (mm)")
        fig.tight_layout()
        fig.savefig("autofocus_sound_speed.png", dpi=120)
        print("wrote autofocus_sound_speed.png")


if __name__ == "__main__":
    main()
