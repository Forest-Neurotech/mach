"""Tests for mach.autograd: the CUDA vector-Jacobian product against PyTorch autograd of the reference DAS."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("mach.autograd needs a CUDA build of torch", allow_module_level=True)

from torch_reference import beamform_reference, simulate_point_scatterers_iq  # noqa: E402  (tests/ is on sys.path)

from mach._cuda_impl import beamform as nb_beamform  # noqa: E402
from mach._cuda_impl import beamform_vjp  # noqa: E402
from mach.autograd import beamform, pad_frames, sharpness  # noqa: E402
from mach.kernel import InterpolationType  # noqa: E402

N_RX, N_SAMPLES, N_SCAN = 32, 256, 300
FS_HZ, F0_HZ, C_M_S = 7.8e6, 7.8e6, 1540.0
DEV = "cuda"
NEAREST, LINEAR = InterpolationType.NearestNeighbor, InterpolationType.Linear


def make_problem(n_frames: int = 8, frame_stride: int = 8):
    """Random I/Q data (frames padded to frame_stride), a linear array at z=0 and a voxel cloud."""
    g = torch.Generator(device=DEV).manual_seed(0)
    chan = torch.zeros((N_RX, N_SAMPLES, frame_stride), dtype=torch.complex64, device=DEV)
    chan[:, :, :n_frames] = torch.complex(
        torch.randn((N_RX, N_SAMPLES, n_frames), generator=g, device=DEV),
        torch.randn((N_RX, N_SAMPLES, n_frames), generator=g, device=DEV),
    )
    rx = torch.zeros((N_RX, 3), device=DEV)
    rx[:, 0] = (torch.arange(N_RX, device=DEV) - (N_RX - 1) / 2) * 0.3e-3
    scan = torch.rand((N_SCAN, 3), generator=g, device=DEV) * torch.tensor([2e-2, 0.0, 2.4e-2], device=DEV)
    scan = scan + torch.tensor([-1e-2, 0.0, 1e-3], device=DEV)
    tx = scan[:, 2] / C_M_S
    return chan, rx, scan, tx


def kwargs(**overrides):
    kw = {
        "rx_start_s": 0.0,
        "sampling_freq_hz": FS_HZ,
        "f_number": 1.0,
        "sound_speed_m_s": C_M_S,
        "modulation_freq_hz": F0_HZ,
        "tukey_alpha": 0.5,
        "interp_type": LINEAR,
    }
    kw.update(overrides)
    return kw


def reference(chan, rx, scan, tx, **kw):
    """Float64 reference; tensors that require grad are passed through as they are."""

    def up(x, dtype):
        return x if x.requires_grad else x.to(dtype)

    return beamform_reference(
        up(chan, torch.complex128), up(rx, torch.float64), up(scan, torch.float64), up(tx, torch.float64), **kw
    )


def rel_norm_err(a, b):
    """||a - b|| / ||b|| in double precision (complex-safe)."""
    wide = torch.complex128 if a.is_complex() else torch.float64
    a, b = a.detach().flatten().to(wide), b.detach().flatten().to(wide)
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def vdot(a, b):
    """<a, b> = sum conj(a) * b in float64."""
    return torch.vdot(a.flatten().to(torch.complex128), b.flatten().to(torch.complex128))


@pytest.mark.parametrize("interp_type", [NEAREST, LINEAR], ids=["nearest", "linear"])
@pytest.mark.parametrize("tukey_alpha", [0.0, 0.5])
@pytest.mark.parametrize("modulation_freq_hz", [F0_HZ, 0.0])
def test_forward_matches_reference(interp_type, tukey_alpha, modulation_freq_hz):
    chan, rx, scan, tx = make_problem()
    kw = kwargs(interp_type=interp_type, tukey_alpha=tukey_alpha, modulation_freq_hz=modulation_freq_hz)
    out = beamform(chan, rx, scan, tx, **kw)
    ref = reference(chan, rx, scan, tx, **kw)
    err = float((out - ref).abs().max() / ref.abs().max())
    print(f"forward max rel err = {err:.2e}")
    assert err < 2e-4


def test_forward_with_padded_frame_axis():
    chan, rx, scan, tx = make_problem(n_frames=6, frame_stride=8)
    out = beamform(chan, rx, scan, tx, n_frames=6, **kwargs())
    ref = reference(chan[:, :, :6].contiguous(), rx, scan, tx, **kwargs())
    assert out.shape == (N_SCAN, 6)
    assert float((out - ref).abs().max() / ref.abs().max()) < 2e-4


@pytest.mark.parametrize("interp_type", [NEAREST, LINEAR], ids=["nearest", "linear"])
@pytest.mark.parametrize("tukey_alpha", [0.0, 0.5])
@pytest.mark.parametrize("modulation_freq_hz", [F0_HZ, 0.0])
def test_adjoint_dot_product(interp_type, tukey_alpha, modulation_freq_hz):
    """<A d, y> == <d, A^H y>: the channel-data VJP is the adjoint of the forward."""
    chan, rx, scan, tx = make_problem()
    kw = kwargs(interp_type=interp_type, tukey_alpha=tukey_alpha, modulation_freq_hz=modulation_freq_hz)
    out = beamform(chan, rx, scan, tx, **kw)
    y = torch.complex(torch.randn_like(out.real), torch.randn_like(out.real))
    grad_chan = torch.zeros_like(chan)
    beamform_vjp(chan, rx, scan, tx, y, grad_channel_data=grad_chan, **kw)
    lhs = vdot(out, y)
    rhs = vdot(chan, grad_chan)
    err = float(abs(lhs - rhs) / abs(lhs))
    print(f"adjoint test |<Ad,y> - <d,A^H y>| / |<Ad,y>| = {err:.2e}")
    assert err < 1e-4


@pytest.mark.parametrize("interp_type", [NEAREST, LINEAR], ids=["nearest", "linear"])
@pytest.mark.parametrize("modulation_freq_hz", [F0_HZ, 0.0])
def test_grad_channel_data_matches_reference(interp_type, modulation_freq_hz):
    chan, rx, scan, tx = make_problem()
    kw = kwargs(interp_type=interp_type, modulation_freq_hz=modulation_freq_hz)
    y = torch.complex(torch.randn(N_SCAN, 8, device=DEV), torch.randn(N_SCAN, 8, device=DEV))

    chan_k = chan.clone().requires_grad_(True)
    torch.real(vdot(y, beamform(chan_k, rx, scan, tx, **kw))).backward()

    chan_r = chan.to(torch.complex128).requires_grad_(True)
    torch.real(vdot(y, reference(chan_r, rx, scan, tx, **kw))).backward()

    err = rel_norm_err(chan_k.grad, chan_r.grad)
    print(f"grad_channel_data rel norm err = {err:.2e}")
    assert err < 1e-3


@pytest.mark.parametrize("interp_type", [NEAREST, LINEAR], ids=["nearest", "linear"])
@pytest.mark.parametrize("modulation_freq_hz", [F0_HZ, 0.0])
def test_grad_geometry_matches_reference(interp_type, modulation_freq_hz):
    """Gradients w.r.t. tx arrivals, scan/rx coordinates, sound speed and rx_start_s."""
    chan, rx, scan, tx = make_problem()
    y = torch.complex(torch.randn(N_SCAN, 8, device=DEV), torch.randn(N_SCAN, 8, device=DEV))

    delays = ((torch.rand(N_RX, device=DEV) - 0.5) * 20e-9).double()  # +-10 ns per-element screen

    def run(fn, dtype):
        # clone: .to() with the same dtype returns the same object, which would alias the fixtures
        rx_ = rx.to(dtype).clone().requires_grad_(True)
        scan_ = scan.to(dtype).clone().requires_grad_(True)
        tx_ = tx.to(dtype).clone().requires_grad_(True)
        delays_ = delays.to(dtype).clone().requires_grad_(True)
        c_ = torch.tensor(C_M_S, dtype=torch.float64, device=DEV, requires_grad=True)
        t0_ = torch.tensor(0.0, dtype=torch.float64, device=DEV, requires_grad=True)
        kw = kwargs(interp_type=interp_type, modulation_freq_hz=modulation_freq_hz, sound_speed_m_s=c_, rx_start_s=t0_)
        torch.real(vdot(y, fn(chan, rx_, scan_, tx_, rx_delays_s=delays_, **kw))).backward()
        return {
            "tx": tx_.grad,
            "scan": scan_.grad,
            "rx": rx_.grad,
            "delays": delays_.grad,
            "c": c_.grad,
            "t0": t0_.grad,
        }

    kernel = run(beamform, torch.float32)
    if interp_type == NEAREST and modulation_freq_hz == 0.0:
        # Neither a phase term nor an interpolant slope: the image is piecewise constant in every
        # delay parameter, so the gradients are identically zero (the reference has no graph at all).
        assert all(float(g.abs().max()) == 0.0 for g in kernel.values())
        return
    ref = run(reference, torch.float64)
    if interp_type == NEAREST:
        # No interpolant slope: rx_start_s only moves the rounding, so the reference has no gradient
        # and the kernel's must be exactly zero.
        assert ref["t0"] is None
        assert float(kernel["t0"]) == 0.0
        del ref["t0"], kernel["t0"]
    errs = {k: rel_norm_err(kernel[k], ref[k]) for k in ref}
    print("geometry grads rel norm err:", {k: f"{v:.2e}" for k, v in errs.items()})
    for name, err in errs.items():
        assert err < 1e-3, f"{name}: {err:.2e}"
    # both are sums of the same dL/dtau(v, e) table, over elements and over voxels respectively
    assert abs(float(kernel["tx"].sum() - kernel["delays"].sum())) < 1e-4 * float(kernel["tx"].abs().sum())


def test_forward_with_rx_delays_matches_reference():
    chan, rx, scan, tx = make_problem()
    delays = (torch.rand(N_RX, device=DEV) - 0.5) * 40e-9
    out = beamform(chan, rx, scan, tx, rx_delays_s=delays, **kwargs())
    ref = reference(chan, rx, scan, tx, rx_delays_s=delays.double(), **kwargs())
    assert float((out - ref).abs().max() / ref.abs().max()) < 2e-4
    plain = beamform(chan, rx, scan, tx, **kwargs())
    assert not torch.allclose(out, plain)


def test_pad_frames():
    chan = torch.zeros((N_RX, N_SAMPLES, 6), dtype=torch.complex64, device=DEV)
    padded, n_frames = pad_frames(chan)
    assert padded.shape == (N_RX, N_SAMPLES, 8) and n_frames == 6 and padded.is_contiguous()
    assert pad_frames(padded)[0].shape[2] == 8


def test_grad_padding_lanes_are_zero():
    chan, rx, scan, tx = make_problem(n_frames=6, frame_stride=8)
    chan.requires_grad_(True)
    beamform(chan, rx, scan, tx, n_frames=6, **kwargs()).abs().sum().backward()
    assert chan.grad[:, :, 6:].abs().max() == 0
    assert chan.grad[:, :, :6].abs().max() > 0


F0_FOCUS_HZ, FS_FOCUS_HZ = 5e6, 20e6


def make_focusing_problem():
    """Point scatterers under a plane wave: the image energy peaks at the true sound speed."""
    g = torch.Generator(device=DEV).manual_seed(3)
    n_rx = 64
    rx = torch.zeros((n_rx, 3), device=DEV)
    rx[:, 0] = (torch.arange(n_rx, device=DEV) - (n_rx - 1) / 2) * 0.3e-3
    scatterers = torch.rand((12, 3), generator=g, device=DEV) * torch.tensor([12e-3, 0.0, 20e-3], device=DEV)
    scatterers = scatterers + torch.tensor([-6e-3, 0.0, 8e-3], device=DEV)
    chan = simulate_point_scatterers_iq(
        rx,
        scatterers,
        torch.ones(12, device=DEV),
        sound_speed_m_s=C_M_S,
        f0_hz=F0_FOCUS_HZ,
        sampling_freq_hz=FS_FOCUS_HZ,
        n_samples=1024,
        n_frames=4,
        pulse_sigma_s=0.5e-6,
    )
    # Sample the image finely enough to resolve the point-spread function (lambda = 0.3 mm, pulse
    # sigma = 0.4 mm): with a coarse grid the energy oscillates as the image shifts across cells.
    x = torch.linspace(-8e-3, 8e-3, 161, device=DEV)
    z = torch.linspace(5e-3, 30e-3, 501, device=DEV)
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    scan = torch.stack([xx.flatten(), torch.zeros_like(xx.flatten()), zz.flatten()], dim=1)
    return chan, rx, scan


def focusing_image(chan, rx, scan, sound_speed_m_s, rx_start_s=0.0):
    """Beamform with the plane-wave transmit arrivals recomputed from the trial sound speed."""
    tx = scan[:, 2] / sound_speed_m_s  # differentiable in sound_speed_m_s through torch as well
    kw = kwargs(
        sampling_freq_hz=FS_FOCUS_HZ,
        modulation_freq_hz=F0_FOCUS_HZ,
        sound_speed_m_s=sound_speed_m_s,
        rx_start_s=rx_start_s,
    )
    return beamform(chan, rx, scan, tx.to(torch.float32), **kw)


def focusing_energy(chan, rx, scan, sound_speed_m_s, rx_start_s=0.0):
    """Image energy: smooth in the delay parameters, used for the finite-difference checks."""
    return focusing_image(chan, rx, scan, sound_speed_m_s, rx_start_s).abs().pow(2).sum()


def focusing_sharpness(chan, rx, scan, sound_speed_m_s):
    return sharpness(focusing_image(chan, rx, scan, sound_speed_m_s))


@pytest.mark.parametrize("parameter", ["sound_speed_m_s", "rx_start_s"])
def test_scalar_gradients_against_finite_differences(parameter):
    """d/dc and d/dt0 of the image energy of a focusing problem: autograd vs central differences.

    Evaluated off-focus (wrong sound speed / start time), where the energy varies strongly. Random
    data cannot be used here: its energy has no systematic dependence on the delays, so the finite
    difference is buried in float32 rounding. Steps keep the delay shift around 1e-2 samples so the
    linear-interpolation kinks and the phase nonlinearity are negligible. (Nearest neighbour is
    piecewise constant in the sample index and cannot be checked this way.)
    """
    chan, rx, scan = make_focusing_problem()
    point = {"sound_speed_m_s": C_M_S + 20.0, "rx_start_s": 0.1e-6}
    step = {"sound_speed_m_s": 0.05, "rx_start_s": 0.5e-9}[parameter]

    def energy(value):
        return focusing_energy(chan, rx, scan, **{**point, parameter: value})

    p = torch.tensor(point[parameter], dtype=torch.float64, device=DEV, requires_grad=True)
    energy(p).backward()
    fd = float(energy(point[parameter] + step) - energy(point[parameter] - step)) / (2 * step)
    err = abs(float(p.grad) - fd) / abs(fd)
    print(f"dE/d{parameter}: autograd = {float(p.grad):.6e}, central FD = {fd:.6e}, rel err = {err:.2e}")
    assert err < 1e-2


def test_autofocus_recovers_sound_speed():
    """Gradient ascent on the image sharpness recovers the true sound speed from 60 m/s away."""
    chan, rx, scan = make_focusing_problem()
    c = torch.tensor(C_M_S + 60.0, dtype=torch.float64, device=DEV, requires_grad=True)
    # Adam moves ~lr per step early on; the decayed learning rates sum to ~100 m/s of travel.
    optimizer = torch.optim.Adam([c], lr=4.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    for _ in range(100):
        optimizer.zero_grad()
        (-focusing_sharpness(chan, rx, scan, c)).backward()
        optimizer.step()
        scheduler.step()
    print(f"autofocus: recovered c = {float(c):.2f} m/s (true {C_M_S})")
    # The sharpness maximum of this finite-aperture, finite-grid problem sits ~1.5 m/s below the
    # true value (a property of the metric, not of the gradient); the descent reaches it.
    assert abs(float(c) - C_M_S) < 4.0


def test_custom_stream_and_no_sync_match_default():
    """Forward and backward launched on a non-default stream without a device synchronize match the default path."""
    chan, rx, scan, tx = make_problem()
    y = torch.complex(torch.randn(N_SCAN, 8, device=DEV), torch.randn(N_SCAN, 8, device=DEV))

    def run():
        chan_ = chan.clone().requires_grad_(True)
        tx_ = tx.clone().requires_grad_(True)
        out = beamform(chan_, rx, scan, tx_, **kwargs())
        torch.real(vdot(y, out)).backward()
        return out.detach().clone(), chan_.grad.clone(), tx_.grad.clone()

    ref = run()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        got = run()
    stream.synchronize()
    for a, b in zip(got, ref, strict=True):
        assert torch.equal(a, b) or rel_norm_err(a, b) < 1e-6
    # the nanobind function with an explicit stream and no synchronize
    out = torch.zeros(N_SCAN, 8, dtype=torch.complex64, device=DEV)
    nb_beamform(chan, rx, scan, tx, out, stream=stream.cuda_stream, synchronize=False, **kwargs())
    stream.synchronize()
    assert rel_norm_err(out, ref[0]) < 1e-6


def test_layout_errors():
    chan, rx, scan, tx = make_problem(n_frames=6, frame_stride=6)
    with pytest.raises(ValueError, match="multiple of 4"):
        beamform(chan, rx, scan, tx, **kwargs())
    chan, rx, scan, tx = make_problem()
    with pytest.raises(ValueError, match="quadratic"):
        beamform(chan, rx, scan, tx, **kwargs(interp_type=InterpolationType.Quadratic))
    with pytest.raises(ValueError, match="CUDA"):
        beamform(chan.cpu(), rx, scan, tx, **kwargs())
    with pytest.raises(TypeError, match="complex64"):
        beamform(chan.real.contiguous(), rx, scan, tx, **kwargs())
    with pytest.raises(RuntimeError, match="multiple of"):
        beamform_vjp(
            chan[:, :, :6].contiguous(),
            rx,
            scan,
            tx,
            torch.zeros(N_SCAN, 6, dtype=torch.complex64, device=DEV),
            grad_channel_data=torch.zeros(N_RX, N_SAMPLES, 6, dtype=torch.complex64, device=DEV),
            **kwargs(),
        )
