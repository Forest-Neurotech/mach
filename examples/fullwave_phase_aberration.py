"""Receive phase-aberration estimation on fullwave-ultra simulations.

A speckle phantom with an anechoic cyst and wire targets is imaged with a 128-element linear
array (plane-wave transmit) through a smooth near-field sound-speed layer, simulated with the
fullwave-ultra 2D solver. The per-element receive delay screen is then estimated from the
aberrated channel data alone by gradient ascent on the image sharpness, using the per-element
delay gradient of ``mach.autograd.beamform`` (``rx_delays_s``).

Ground truth comes two ways: the straight-ray travel-time integral through the layer, and the
arrival-time profile measured on a lone wire target simulated through the same layer (which also
carries the transmit-side distortion, so it is the more honest reference). Four solver runs:

    control      phantom, no layer
    aberrated    phantom + layer
    point_ctrl   lone wire target, no layer      (removes systematic delay bias)
    point_aber   lone wire target + layer        -> measured screen

Requires ``fullwave2_ultra`` (with its solver binaries) and a CUDA build of torch::

    python examples/fullwave_phase_aberration.py --out-dir results/fullwave_aberration [--reuse]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from fullwave2_ultra import io_dat, sim, solver, stability
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, hilbert, sosfiltfilt

from mach.autograd import beamform

# ----------------------------------------------------------------------------- configuration
C0 = 1540.0
F0 = 5e6
LAMBDA = C0 / F0
PPW = 10.0
DX = LAMBDA / PPW  # 30.8 um
N_RX = 128
PITCH_CELLS = 10  # 0.308 mm = lambda
ELEMENT_CELLS = 8  # active cells per element (kerf 2)
WIDTH_M, DEPTH_M = 44e-3, 42e-3
J_SRC = 3  # 1-based interior row of the transmit line
RX_OFFSET = 2  # receiver row = J_SRC + RX_OFFSET (off the hard-set source row)
MOD_T = 4
N_CYCLES_CENTER = 2.5  # pulse centre, in periods
LAYERS = {
    # name: (z_top, z_bottom, axial sigma). "thin" is a near-field phase screen at the face (the
    # model the estimator assumes); "thick" sits deeper, so rays from different voxels cross it at
    # different places and no single receive screen is exact. The lateral structure is set by the
    # coherence length (autocorrelation FWHM): a Gaussian filter of sigma_f gives an autocorrelation
    # FWHM of 2*sqrt(2 ln 2)*sqrt(2)*sigma_f = 3.33*sigma_f.
    "thin": (0.3e-3, 2.3e-3, 0.5e-3),
    "thick": (1.5e-3, 7.5e-3, 1.0e-3),
}
COHERENCE_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)) * math.sqrt(2.0))


def coherence_length_m(screen, x):
    """Realised coherence length: FWHM of the screen's lateral autocorrelation (linear interpolation)."""
    v = screen - screen.mean()
    ac = np.correlate(v, v, mode="full")[v.size - 1 :]
    ac = ac / ac[0]
    dx = float(np.mean(np.diff(x)))
    below = np.nonzero(ac < 0.5)[0]
    if below.size == 0:
        return float("nan")
    k = below[0]
    frac = (ac[k - 1] - 0.5) / (ac[k - 1] - ac[k])
    return 2.0 * (k - 1 + frac) * dx


CYST = (5e-3, 25e-3, 3e-3)  # x, z, radius
WIRES = [(0.0, 10e-3), (0.0, 20e-3), (0.0, 30e-3), (-6e-3, 15e-3), (-6e-3, 30e-3)]
LONE_WIRE = (0.0, 25e-3)
DEV = "cuda"


def grid_shape():
    return round(WIDTH_M / DX), round(DEPTH_M / DX)


def element_columns():
    """1-based interior column indices of every element's active cells and centres (n_rx, cells)."""
    n_x, _ = grid_shape()
    aperture = N_RX * PITCH_CELLS
    i0 = (n_x - aperture) // 2 + 1 + (PITCH_CELLS - ELEMENT_CELLS) // 2
    cols = i0 + np.arange(N_RX)[:, None] * PITCH_CELLS + np.arange(ELEMENT_CELLS)[None, :]
    return cols


def rx_coords_m():
    cols = element_columns()
    n_x, _ = grid_shape()
    x = (cols.mean(1) - (n_x + 1) / 2) * DX
    return np.stack([x, np.zeros(N_RX), np.zeros(N_RX)], 1).astype(np.float32)


def z_of_row(j):
    """Depth (m) of 1-based interior row j, measured from the receiver row."""
    return (j - (J_SRC + RX_OFFSET)) * DX


def x_of_col(i):
    n_x, _ = grid_shape()
    return (i - (n_x + 1) / 2) * DX


# ----------------------------------------------------------------------------- phantom / layer
def base_maps(rng, *, speckle: bool, wires, cyst):
    n_x, n_y = grid_shape()
    cmap = np.full((n_x, n_y), C0)
    rmap = np.full((n_x, n_y), 1000.0)
    ii, jj = np.meshgrid(np.arange(1, n_x + 1), np.arange(1, n_y + 1), indexing="ij")
    xx, zz = x_of_col(ii), z_of_row(jj)
    if speckle:
        # ~10% of cells carry a +-2% sound-speed perturbation: ~10 scatterers per resolution cell
        scat = np.where(rng.random((n_x, n_y)) < 0.10, rng.uniform(-1.0, 1.0, (n_x, n_y)), 0.0)
        scat[zz < 0.5e-3] = 0.0
        if cyst is not None:
            cx, cz, cr = cyst
            scat[(xx - cx) ** 2 + (zz - cz) ** 2 < cr**2] = 0.0
        cmap = cmap * (1.0 + 0.02 * scat)
    for wx, wz in wires:
        rmap[(xx - wx) ** 2 + (zz - wz) ** 2 <= (1.5 * DX) ** 2] = 2500.0  # 3-cell wire, impedance x2.5
    return cmap, rmap


def aberrating_layer(rng, layer, target_rms_s=50e-9, coherence_m=5e-3):
    """Smooth random sound-speed perturbation in a near-field slab with the requested lateral
    coherence length (autocorrelation FWHM), scaled so that the straight-ray one-way delay screen
    has the requested RMS. Returns (delta_c map, screen)."""
    n_x, n_y = grid_shape()
    jj = np.arange(1, n_y + 1)
    zz = z_of_row(jj)
    z0, z1, sigma_z = LAYERS[layer]
    sigma_x = coherence_m * COHERENCE_TO_SIGMA
    noise = rng.standard_normal((n_x, n_y))
    field = gaussian_filter(noise, sigma=(sigma_x / DX, sigma_z / DX), mode="reflect")
    field = field / field.std()
    ramp = min(1.0e-3, 0.25 * (z1 - z0))
    window = np.clip((zz - z0) / ramp, 0, 1) * np.clip((z1 - zz) / ramp, 0, 1)
    window = np.sin(0.5 * np.pi * window) ** 2
    delta_c = field * window[None, :]
    screen = -(delta_c / C0**2 * DX).sum(1)  # one-way straight-ray delay per column, per unit delta_c
    cols = element_columns() - 1
    scale = target_rms_s / screen[cols].mean(1).std()
    delta_c *= scale
    screen = -(delta_c / C0**2 * DX).sum(1)
    return delta_c, screen[cols].mean(1)


# ----------------------------------------------------------------------------- solver runs
def pulse(t):
    period = 1.0 / F0
    arg = t - N_CYCLES_CENTER * period
    return np.sin(2 * np.pi * F0 * arg) * np.exp(-((arg / period) ** 2))


def run_solver(rundir, cmap, rmap, *, reuse=False):
    """Write a run dir, run bench_2d_batch, return (rf (n_rx, nframes), fs, meta)."""
    n_x, n_y = grid_shape()
    cols = element_columns()
    incoords = np.stack([cols.ravel(), np.full(cols.size, J_SRC)], 1)
    outcoords = np.stack([cols.ravel(), np.full(cols.size, J_SRC + RX_OFFSET)], 1)
    xdc = {"incoords": incoords, "outcoords": outcoords}
    genout = os.path.join(rundir, "genout.dat")
    if reuse and os.path.exists(genout):
        d_t = io_dat.read_float(os.path.join(rundir, "dT.dat"))
        n_t = io_dat.read_int(os.path.join(rundir, "nT.dat"))
        n_tic = io_dat.read_int(os.path.join(rundir, "nTic.dat"))
        cfl = d_t * C0 / DX
    else:
        cfl = 0.4
        limit = stability.cfl_limit(M=8, dim=2)
        cfl = min(cfl, 0.95 * limit * C0 / cmap.max())
        dur = (2 * DEPTH_M) / C0 + 4e-6
        omega0 = 2 * math.pi * F0
        n_t = round(dur * F0 * PPW / cfl)
        d_t = DX / C0 * cfl
        n_tic = math.ceil(2 * N_CYCLES_CENTER / F0 / d_t)
        maps = {"cmap": cmap, "rmap": rmap, "nmap": np.zeros_like(cmap), "amap": np.full_like(cmap, 0.5)}
        meta = sim.write_fullwave_sim(
            rundir, C0, omega0, dur, PPW, cfl, maps, xdc, n_tic, MOD_T, source_zero_window=(n_t, n_t)
        )
        assert meta["nT"] == n_t and abs(meta["dT"] - d_t) < 1e-15, (meta, n_t, d_t)
        t = np.arange(n_tic) * d_t
        io_dat.write_icmat(os.path.join(rundir, "icmat.dat"), [np.tile(pulse(t)[None, :] * 1e5, (cols.size, 1))])
        io_dat.write_int(os.path.join(rundir, "nsims.dat"), 1)
        t_start = time.time()
        solver.run(rundir, name="bench_2d_batch", capture_output=True)
        print(f"  solver {os.path.basename(rundir)}: {n_x}x{n_y} interior, nT={n_t}, {time.time() - t_start:.1f} s")
    g = io_dat.read_genout(genout, cols.size)  # (nframes, ncoordsout)
    rf = g.reshape(g.shape[0], N_RX, ELEMENT_CELLS).mean(2).T  # (n_rx, nframes)
    fs = 1.0 / (d_t * MOD_T)
    return rf.astype(np.float64), fs, {"dT": d_t, "nT": n_t, "nTic": n_tic, "cfl": cfl}


# ----------------------------------------------------------------------------- channel data
def bandpass(rf, fs, bw=0.7):
    sos = butter(4, [F0 * (1 - bw / 2), F0 * (1 + bw / 2)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, rf, axis=1)


def transmit_peak_frame(rf, fs):
    """Frame index of the outgoing pulse as seen by the receivers (per-element median)."""
    n = int(3 * N_CYCLES_CENTER / F0 * fs)
    env = np.abs(hilbert(rf[:, :n], axis=1))
    return float(np.median(env.argmax(1)))


def to_iq(rf, fs):
    """Baseband I/Q (n_rx, n_samples, 4) complex64 (frame axis padded to 4 for the kernel)."""
    t = np.arange(rf.shape[1]) / fs
    iq = hilbert(rf, axis=1) * np.exp(-2j * np.pi * F0 * t)[None, :]
    chan = np.zeros((N_RX, rf.shape[1], 4), np.complex64)
    chan[:, :, 0] = iq
    return torch.as_tensor(chan, device=DEV)


# ----------------------------------------------------------------------------- imaging
def scan_grid(x_mm=(-15, 15, 0.1), z_mm=(3, 38, 0.05)):
    x = np.arange(x_mm[0], x_mm[1] + 1e-9, x_mm[2]) * 1e-3
    z = np.arange(z_mm[0], z_mm[1] + 1e-9, z_mm[2]) * 1e-3
    xx, zz = np.meshgrid(x, z, indexing="ij")
    scan = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], 1).astype(np.float32)
    return x, z, torch.as_tensor(scan, device=DEV)


def image(chan, rx, scan, rx_start_s, delays=None, fs=None):
    tx = (scan[:, 2] / C0).to(torch.float32)
    out = beamform(
        chan,
        rx,
        scan,
        tx,
        rx_start_s=rx_start_s,
        sampling_freq_hz=fs,
        f_number=1.0,
        sound_speed_m_s=C0,
        modulation_freq_hz=F0,
        tukey_alpha=0.5,
        rx_delays_s=delays,
        n_frames=1,
    )
    return out[:, 0]


def sharpness(img, mask):
    intensity = img.abs().pow(2)[mask]
    return intensity.pow(2).sum() / intensity.sum().pow(2)


def envelope_db(img, nx, nz):
    env = img.detach().abs().reshape(nx, nz).T.cpu().numpy()
    return 20 * np.log10(env / env.max() + 1e-12)


def cyst_contrast_db(env, x, z, cyst=CYST):
    cx, cz, cr = cyst
    xx, zz = np.meshgrid(x, z, indexing="xy")
    inside = (xx - cx) ** 2 + (zz - cz) ** 2 < (0.7 * cr) ** 2
    ring = ((xx - cx) ** 2 + (zz - cz) ** 2 > (1.3 * cr) ** 2) & ((xx - cx) ** 2 + (zz - cz) ** 2 < (2.0 * cr) ** 2)
    return 20 * np.log10(env[inside].mean() / env[ring].mean())


def wire_fwhm_mm(env, x, z, wire):
    """Lateral FWHM (mm) and peak of a wire: lateral profile through the brightest pixel within
    +-0.5 mm axially and +-2 mm laterally of the nominal position (keeps neighbouring wires out)."""
    wx, wz = wire
    zi = np.argmin(np.abs(z - wz))
    xs = np.nonzero(np.abs(x - wx) < 2e-3)[0]
    band = env[max(zi - 10, 0) : zi + 11][:, xs]
    iz, _ = np.unravel_index(band.argmax(), band.shape)
    profile = band[iz]
    half = profile.max() / 2
    above = np.nonzero(profile >= half)[0]
    return (x[xs][above[-1]] - x[xs][above[0]]) * 1e3, profile.max()


# ----------------------------------------------------------------------------- measured screen
def measured_delays(rf, fs, rx, peak_frame, wire=LONE_WIRE):
    """Per-element arrival-time deviation of the lone wire's echo from the geometric delay,
    by cross-correlation with the centre element (sub-sample parabolic peak)."""
    wx, wz = wire
    geo = (wz + np.hypot(rx[:, 0] - wx, wz)) / C0  # (n_rx,)
    t = (np.arange(rf.shape[1]) - peak_frame) / fs
    half = 1.5e-6
    ref = N_RX // 2
    lag = np.zeros(N_RX)
    for e in range(N_RX):
        w_e = (t > geo[e] - half) & (t < geo[e] + half)
        w_r = (t > geo[ref] - half) & (t < geo[ref] + half)
        a, b = rf[e, w_e], rf[ref, w_r]
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        xc = np.correlate(a, b, mode="full")
        k = xc.argmax()
        if 0 < k < xc.size - 1:
            y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        lag[e] = (k - (n - 1)) / fs  # a lags b by `lag` beyond the geometric alignment
    return lag  # relative to the centre element


def detrend(v, x):
    a = np.polyfit(x, v, 1)
    return v - np.polyval(a, x)


HIGHPASS_SIGMA_ELEMENTS = 16.0  # ~5 mm: components smoother than the coherence length are removed


def highpass(v, sigma_elements=HIGHPASS_SIGMA_ELEMENTS):
    """Remove the smooth part of a screen (Gaussian, replicate edges). Image-domain objectives are
    largely blind to receive-delay components smoother than the receive aperture, which only shift
    or refocus features; comparing at the coherence-length scale keeps the screen's own structure."""
    return v - gaussian_filter(v, sigma_elements, mode="nearest")


class SmoothScreen(torch.nn.Module):
    """Per-element delays parametrised through a Gaussian smoother of ``sigma`` elements: a
    smoothness prior matching the physical screen, which keeps the sharpness objective from
    growing element-to-element alternating patterns that raise sharpness but destroy contrast."""

    def __init__(self, n, sigma, init=None):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(n, device=DEV) if init is None else init.clone())
        k = torch.arange(-int(3 * sigma), int(3 * sigma) + 1, device=DEV, dtype=torch.float32)
        k = torch.exp(-0.5 * (k / sigma) ** 2)
        self.register_buffer("kernel", (k / k.sum()).view(1, 1, -1))

    def forward(self):
        pad = self.kernel.shape[-1] // 2
        x = torch.nn.functional.pad(self.p.view(1, 1, -1), (pad, pad), mode="replicate")
        return torch.nn.functional.conv1d(x, self.kernel).view(-1)


# ----------------------------------------------------------------------------- main
def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="results/fullwave_aberration")
    parser.add_argument("--rms-ns", type=float, default=50.0, help="target RMS of the straight-ray screen")
    parser.add_argument(
        "--coherence-mm", type=float, default=5.0, help="lateral coherence length of the screen (autocorrelation FWHM)"
    )
    parser.add_argument("--reuse", action="store_true", help="reuse existing solver outputs in out-dir")
    parser.add_argument("--iterations", type=int, default=150, help="iterations per stage")
    parser.add_argument(
        "--sigma-stages", default="8,3", help="coarse-to-fine smoothing widths (elements) of the screen parametrisation"
    )
    parser.add_argument("--layer", choices=sorted(LAYERS), default="thin")
    parser.add_argument("--objective", choices=["sharpness", "energy"], default="sharpness")
    parser.add_argument(
        "--roi",
        choices=["full", "wire30"],
        default="full",
        help="region the objective is evaluated on: the whole image below the layer, "
        "or a 10 x 6 mm patch around the (0, 30 mm) wire (an isoplanatic patch)",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    n_x, n_y = grid_shape()
    print(f"grid {n_x}x{n_y} interior at {DX * 1e6:.1f} um, {N_RX} elements, pitch {PITCH_CELLS * DX * 1e3:.3f} mm")
    delta_c, screen_ray = aberrating_layer(
        np.random.default_rng(1), args.layer, args.rms_ns * 1e-9, args.coherence_mm * 1e-3
    )
    layer_z = LAYERS[args.layer][:2]
    print(
        f"layer: delta c in [{delta_c.min():+.0f}, {delta_c.max():+.0f}] m/s, straight-ray screen RMS {screen_ray.std() * 1e9:.1f} ns, "
        f"coherence length {coherence_length_m(screen_ray, rx_coords_m()[:, 0]) * 1e3:.1f} mm (target {args.coherence_mm:.1f})"
    )

    cmap_ph, rmap_ph = base_maps(rng, speckle=True, wires=WIRES, cyst=CYST)
    cmap_pt, rmap_pt = base_maps(rng, speckle=False, wires=[LONE_WIRE], cyst=None)
    runs = {
        "control": (cmap_ph, rmap_ph),
        "aberrated": (cmap_ph + delta_c, rmap_ph),
        "point_ctrl": (cmap_pt, rmap_pt),
        "point_aber": (cmap_pt + delta_c, rmap_pt),
    }
    rf, fs, meta = {}, None, None
    for name, (cmap, rmap) in runs.items():
        rf[name], fs, meta = run_solver(os.path.join(args.out_dir, name), cmap, rmap, reuse=args.reuse)
    print(
        f"fs = {fs / 1e6:.1f} MHz, {rf['control'].shape[1]} samples, dT = {meta['dT'] * 1e9:.2f} ns, cfl = {meta['cfl']:.3f}"
    )

    rx_np = rx_coords_m()
    rx = torch.as_tensor(rx_np, device=DEV)
    peak = transmit_peak_frame(rf["control"], fs)
    rx_start_s = -peak / fs
    for k in rf:
        rf[k] = bandpass(rf[k], fs)
    chan = {k: to_iq(v, fs) for k, v in rf.items()}

    # measured screen from the lone wire (aberrated minus control removes geometric bias)
    lag_aber = measured_delays(rf["point_aber"], fs, rx_np, peak)
    lag_ctrl = measured_delays(rf["point_ctrl"], fs, rx_np, peak)
    screen_meas = lag_aber - lag_ctrl
    screen_meas -= screen_meas.mean()
    screen_ray_c = screen_ray - screen_ray.mean()
    print(
        f"measured screen RMS {screen_meas.std() * 1e9:.1f} ns; straight-ray vs measured RMS diff "
        f"{(screen_meas - screen_ray_c).std() * 1e9:.1f} ns (after detrend: {detrend(screen_meas - screen_ray_c, rx_np[:, 0]).std() * 1e9:.1f} ns)"
    )

    # imaging + estimation
    x, z, scan = scan_grid()
    nx, nz = x.size, z.size
    scan_np = scan.cpu().numpy()
    below_layer = scan_np[:, 2] > layer_z[1] + 1.5e-3
    if args.roi == "wire30":
        below_layer &= (np.abs(scan_np[:, 0]) < 5e-3) & (np.abs(scan_np[:, 2] - 30e-3) < 3e-3)
    mask = torch.as_tensor(below_layer, device=DEV)
    objective = sharpness if args.objective == "sharpness" else (lambda img, m: img.abs().pow(2)[m].sum() * 1e-9)
    zero = torch.zeros(N_RX, device=DEV)
    with torch.no_grad():
        img = {
            "control": image(chan["control"], rx, scan, rx_start_s, zero, fs),
            "aberrated": image(chan["aberrated"], rx, scan, rx_start_s, zero, fs),
            "measured": image(
                chan["aberrated"],
                rx,
                scan,
                rx_start_s,
                torch.as_tensor(screen_meas, device=DEV, dtype=torch.float32),
                fs,
            ),
        }
    history = []
    t_start = time.time()
    init = None
    for stage, sigma in enumerate(float(v) for v in args.sigma_stages.split(",")):
        screen = SmoothScreen(N_RX, sigma, init)  # delays in ns
        opt = torch.optim.Adam(screen.parameters(), lr=3.0 if stage == 0 else 1.0)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
        for it in range(args.iterations):
            opt.zero_grad()
            s = objective(image(chan["aberrated"], rx, scan, rx_start_s, screen() * 1e-9, fs), mask)
            (-s).backward()
            est = screen().detach().cpu().numpy() * 1e-9
            resid = highpass(est - screen_meas).std()
            history.append((float(s), resid))
            if it % 50 == 0 or it == args.iterations - 1:
                print(
                    f"  stage {stage} (sigma {sigma:g}) it {it:3d}: {args.objective} {float(s):.3e}  "
                    f"high-pass residual vs measured {resid * 1e9:5.1f} ns"
                )
            opt.step()
            sched.step()
        init = screen().detach()  # carry the realised screen (not the raw parameters) into the finer stage
    print(f"estimation: {len(history)} iterations in {time.time() - t_start:.1f} s")
    est = screen().detach() * 1e-9
    with torch.no_grad():
        img["estimated"] = image(chan["aberrated"], rx, scan, rx_start_s, est.float(), fs)
        sharp = {k: float(sharpness(v, mask)) for k, v in img.items()}
        # objective landscape along the measured-screen direction (alpha * measured screen)
        landscape = {}
        for alpha in (-0.5, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
            im_a = image(
                chan["aberrated"],
                rx,
                scan,
                rx_start_s,
                torch.as_tensor(alpha * screen_meas, device=DEV, dtype=torch.float32),
                fs,
            )
            landscape[f"{alpha:+.2f}"] = {
                "sharpness": float(sharpness(im_a, mask)),
                "energy": float(im_a.abs().pow(2)[mask].sum()),
            }
    est_np = est.cpu().numpy()
    est_np -= est_np.mean()

    # metrics
    env = {k: np.abs(v.reshape(nx, nz).T.cpu().numpy()) for k, v in img.items()}
    metrics = {
        "screen_rms_ns": {
            "straight_ray": screen_ray_c.std() * 1e9,
            "measured": screen_meas.std() * 1e9,
            "estimated": est_np.std() * 1e9,
        },
        "coherence_length_mm": {
            "target": args.coherence_mm,
            "straight_ray": coherence_length_m(screen_ray_c, rx_np[:, 0]) * 1e3,
            "measured": coherence_length_m(screen_meas, rx_np[:, 0]) * 1e3,
            "estimated": coherence_length_m(est_np, rx_np[:, 0]) * 1e3,
        },
        "residual_rms_ns": {
            "est_vs_measured": (est_np - screen_meas).std() * 1e9,
            "est_vs_measured_detrended": detrend(est_np - screen_meas, rx_np[:, 0]).std() * 1e9,
            "est_vs_measured_highpass": highpass(est_np - screen_meas).std() * 1e9,
            "measured_highpass_rms": highpass(screen_meas).std() * 1e9,
            "highpass_correlation": float(np.corrcoef(highpass(est_np), highpass(screen_meas))[0, 1]),
            "raw_correlation": float(np.corrcoef(est_np, screen_meas)[0, 1]),
            "regression_slope": float(np.dot(est_np, screen_meas) / np.dot(screen_meas, screen_meas)),
            "est_vs_ray_detrended": detrend(est_np - screen_ray_c, rx_np[:, 0]).std() * 1e9,
            "measured_vs_ray_detrended": detrend(screen_meas - screen_ray_c, rx_np[:, 0]).std() * 1e9,
        },
        "sharpness": sharp,
        "landscape_along_measured_screen": landscape,
        "config": {
            "layer": args.layer,
            "layer_z_mm": [1e3 * v for v in layer_z],
            "objective": args.objective,
            "roi": args.roi,
            "rms_ns": args.rms_ns,
            "iterations": args.iterations,
            "sigma_stages": args.sigma_stages,
            "coherence_mm": args.coherence_mm,
        },
        "cyst_contrast_db": {},
        "wire_fwhm_mm": {},
        "wire_peak_rel": {},
    }
    for k, e in env.items():
        metrics["cyst_contrast_db"][k] = cyst_contrast_db(e, x, z)
        metrics["wire_fwhm_mm"][k] = {
            f"{wx * 1e3:.0f},{wz * 1e3:.0f}": wire_fwhm_mm(e, x, z, (wx, wz))[0] for wx, wz in WIRES
        }
        metrics["wire_peak_rel"][k] = {
            f"{wx * 1e3:.0f},{wz * 1e3:.0f}": wire_fwhm_mm(e, x, z, (wx, wz))[1]
            / wire_fwhm_mm(env["control"], x, z, (wx, wz))[1]
            for wx, wz in WIRES
        }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    np.savez(
        os.path.join(args.out_dir, "screens.npz"),
        x_m=rx_np[:, 0],
        straight_ray=screen_ray_c,
        measured=screen_meas,
        estimated=est_np,
        history=np.array(history),
    )
    for key in ("screen_rms_ns", "coherence_length_mm", "residual_rms_ns", "sharpness", "cyst_contrast_db"):
        print(f"  {key}: " + ", ".join(f"{k} {float(v):.3g}" for k, v in metrics[key].items()))
    print(
        "  wire FWHM (mm): "
        + ", ".join(
            f"{k} {metrics['wire_fwhm_mm'][k]['0,20']:.2f}/{metrics['wire_fwhm_mm'][k]['0,30']:.2f}" for k in env
        )
    )
    print(
        "  wire peak rel. control: "
        + ", ".join(
            f"{k} {metrics['wire_peak_rel'][k]['0,20']:.2f}/{metrics['wire_peak_rel'][k]['0,30']:.2f}" for k in env
        )
    )

    # figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
    titles = {
        "control": "no aberrating layer",
        "aberrated": "aberrated, uncorrected",
        "estimated": "corrected with estimated screen",
        "measured": "corrected with measured screen",
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.8), sharey=True)
    for ax, k in zip(axes, ("control", "aberrated", "estimated", "measured"), strict=True):
        ax.imshow(
            20 * np.log10(env[k] / env[k].max() + 1e-12), extent=extent, cmap="gray", vmin=-55, vmax=0, aspect="equal"
        )
        ax.set_title(
            f"{titles[k]}\ncyst contrast {metrics['cyst_contrast_db'][k]:.1f} dB, "
            f"wire(0,20) FWHM {metrics['wire_fwhm_mm'][k]['0,20']:.2f} mm",
            fontsize=10,
        )
        ax.set_xlabel("x (mm)")
        ax.axhspan(layer_z[0] * 1e3, layer_z[1] * 1e3, color="tab:orange", alpha=0.12, lw=0)
    axes[0].set_ylabel("z (mm)")
    fig.suptitle(
        f"fullwave-ultra phantom, {args.layer} aberrating layer at {layer_z[0] * 1e3:.1f}-{layer_z[1] * 1e3:.1f} mm "
        f"({args.rms_ns:.0f} ns RMS, {args.coherence_mm:.0f} mm coherence length); receive-only correction, {args.objective} objective, ROI {args.roi}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out_dir, "bmode.png"), dpi=110)

    xe = rx_np[:, 0] * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.plot(
        xe,
        screen_ray_c * 1e9,
        color="0.5",
        label=f"straight-ray integral (RMS {screen_ray_c.std() * 1e9:.0f} ns, coherence {metrics['coherence_length_mm']['straight_ray']:.1f} mm)",
    )
    ax.plot(
        xe,
        screen_meas * 1e9,
        color="k",
        label=f"measured on lone wire (RMS {screen_meas.std() * 1e9:.0f} ns, coherence {metrics['coherence_length_mm']['measured']:.1f} mm)",
    )
    ax.plot(xe, est_np * 1e9, label=f"estimated by sharpness ascent (RMS {est_np.std() * 1e9:.0f} ns)", color="tab:red")
    ax.set_xlabel("element x (mm)")
    ax.set_ylabel("receive delay (ns)")
    ax.set_title("receive delay screen (mean removed)")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.plot(
        xe,
        highpass(screen_meas) * 1e9,
        color="k",
        label=f"measured, high-pass (RMS {metrics['residual_rms_ns']['measured_highpass_rms']:.0f} ns)",
    )
    ax.plot(
        xe,
        highpass(est_np) * 1e9,
        color="tab:red",
        label=f"estimated, high-pass (residual RMS {metrics['residual_rms_ns']['est_vs_measured_highpass']:.1f} ns, "
        f"corr {metrics['residual_rms_ns']['highpass_correlation']:.2f})",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("element x (mm)")
    ax.set_ylabel("delay (ns)")
    ax.set_title(
        f"components smoother than {HIGHPASS_SIGMA_ELEMENTS * PITCH_CELLS * DX * 1e3:.0f} mm removed from both\n"
        f"(raw screens: correlation {metrics['residual_rms_ns']['raw_correlation']:.2f}, regression slope {metrics['residual_rms_ns']['regression_slope']:.2f})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    h = np.array(history)
    ax.plot(h[:, 0], color="tab:blue")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sharpness", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(h[:, 1] * 1e9, color="tab:red")
    ax2.set_ylabel("high-pass residual RMS vs measured (ns)", color="tab:red")
    ax.set_title("convergence")
    ax = axes[1, 1]
    jj = np.arange(1, n_y + 1)
    zl = z_of_row(jj) * 1e3
    sel = zl < layer_z[1] * 1e3 + 2.5
    im = ax.imshow(
        delta_c[:, sel].T,
        extent=[x_of_col(1) * 1e3, x_of_col(n_x) * 1e3, zl[sel][-1], zl[sel][0]],
        cmap="RdBu_r",
        vmin=-np.abs(delta_c).max(),
        vmax=np.abs(delta_c).max(),
        aspect="auto",
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("aberrating layer: sound-speed deviation (m/s)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "screen.png"), dpi=110)
    print(f"wrote {args.out_dir}/bmode.png, screen.png, metrics.json, screens.npz")


if __name__ == "__main__":
    main()
