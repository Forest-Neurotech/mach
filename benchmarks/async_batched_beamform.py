"""Compare synchronous and asynchronous Mach with one or two device slots.

Run with CuPy and PyMUST installed:
    python benchmarks/async_batched_beamform.py --output async-benchmark.json
"""

import argparse
import hashlib
import itertools
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import cupy as cp
import numpy as np
import pymust

from mach.io.must import download_pymust_doppler_data, extract_pymust_params, linear_probe_positions, scan_grid
from mach.kernel import beamform

# Match the PyMUST rotating-disk grid in the standard benchmark (tests/compare/conftest.py).
PYMUST_LATERAL_BOUNDS_M = (-0.0125, 0.0125)
PYMUST_ELEVATION_M = 0.0
PYMUST_DEPTH_BOUNDS_M = (0.010, 0.035)
PYMUST_GRID_POINTS_PER_AXIS = 251


def run_batches(batches, geometry, slots, synchronize, *, capture=False):
    """Copy from pinned host buffers, beamform, and reduce on the default stream."""
    compute = cp.cuda.Stream.null
    copy = cp.cuda.Stream(non_blocking=True)
    device = [cp.empty(batches[0].shape, dtype=batches[0].dtype) for _ in range(slots)]
    points = geometry["scan_coords_m"].shape[0]
    frames = batches[0].shape[-1]
    outputs = [cp.empty((points, frames), dtype=cp.complex64) for _ in range(slots)]
    powers = [cp.empty(points, dtype=cp.float32) for _ in range(slots)]
    started = [cp.cuda.Event() for _ in batches]
    copied = [cp.cuda.Event(disable_timing=True) for _ in batches]
    finished = [cp.cuda.Event() for _ in batches]
    snapshots = cp.empty((len(batches), points, frames), dtype=cp.complex64) if capture else None

    def upload(index):
        slot = index % slots
        if index >= slots:
            copy.wait_event(finished[index - slots])
        batch = batches[index]
        started[index].record(copy)
        cp.cuda.runtime.memcpyAsync(
            device[slot].data.ptr,
            batch.ctypes.data,
            batch.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
            copy.ptr,
        )
        copied[index].record(copy)

    with compute:
        compute.synchronize()
        start = time.perf_counter()
        upload(0)
        if slots == 2 and len(batches) > 1:
            upload(1)
        for index in range(len(batches)):
            slot = index % slots
            compute.wait_event(copied[index])
            out = outputs[slot]
            out.fill(0)
            result = beamform(device[slot], out=out, synchronize=synchronize, **geometry)
            cp.sum(cp.abs(result) ** 2, axis=-1, out=powers[slot])
            if snapshots is not None:
                cp.copyto(snapshots[index], result)
            finished[index].record(compute)
            # For two slots this submits the copy after next, into the slot
            # just used. For one slot it submits the next copy into that slot.
            if index + slots < len(batches):
                upload(index + slots)
        compute.synchronize()
        elapsed = time.perf_counter() - start

    service_ms = [cp.cuda.get_elapsed_time(first, last) for first, last in zip(started, finished, strict=True)]
    return {
        "seconds": elapsed,
        "completed_frames_per_s": frames * len(batches) / elapsed,
        "copy_start_to_power_done_p95_ms": float(np.percentile(service_ms, 95)),
    }, snapshots


def load_phantom(batch_size):
    """Prepare distinct frame batches and keep geometry on the device."""
    data = download_pymust_doppler_data()
    params = extract_pymust_params(data)
    iq = pymust.rf2iq(data["RF"].astype(float), params).transpose(1, 0, 2)
    if iq.shape[-1] % batch_size:
        raise ValueError("Batch size must divide the recorded frame count.")
    batches = []
    for first in range(0, iq.shape[-1], batch_size):
        chunk = np.ascontiguousarray(iq[..., first : first + batch_size], dtype=np.complex64)
        owner = cp.cuda.alloc_pinned_memory(chunk.nbytes)
        batch = np.frombuffer(owner, dtype=chunk.dtype, count=chunk.size).reshape(chunk.shape)
        np.copyto(batch, chunk)
        batches.append(batch)

    with cp.cuda.Stream.null:
        grid = scan_grid(
            np.linspace(*PYMUST_LATERAL_BOUNDS_M, num=PYMUST_GRID_POINTS_PER_AXIS),
            np.array([PYMUST_ELEVATION_M]),
            np.linspace(*PYMUST_DEPTH_BOUNDS_M, num=PYMUST_GRID_POINTS_PER_AXIS),
        )
        geometry = {
            "rx_coords_m": cp.asarray(linear_probe_positions(params["Nelements"], params["pitch"]), dtype=cp.float32),
            "scan_coords_m": cp.asarray(grid, dtype=cp.float32),
            "tx_wave_arrivals_s": cp.asarray(grid[:, 2] / params["c"], dtype=cp.float32),
            "rx_start_s": float(params["t0"]),
            "sampling_freq_hz": float(params["fs"]),
            "f_number": float(params["fnumber"]),
            "sound_speed_m_s": float(params["c"]),
            "modulation_freq_hz": float(params["fc"]),
            "tukey_alpha": 0.0,
        }
    return batches, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--slots", type=int, choices=(1, 2))
    parser.add_argument("--mode", choices=("sync", "async"))
    parser.add_argument("--verify-only", action="store_true", help="Check parity without timing trials.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cycles < 1 or args.repeats < 1:
        parser.error("--cycles and --repeats must be positive")

    batches, geometry = load_phantom(args.batch_size)
    conditions = list(
        itertools.product((args.slots,) if args.slots else (1, 2), (args.mode,) if args.mode else ("sync", "async"))
    )
    trials = {f"{slots}-slot-{mode}": [] for slots, mode in conditions}
    checks = {}
    verification_batches = batches * 3
    _, reference = run_batches(verification_batches, geometry, 1, True, capture=True)
    for slots, mode in conditions:
        _, actual = run_batches(verification_batches, geometry, slots, mode == "sync", capture=True)
        cp.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-4)
        checks[f"{slots}-slot-{mode}"] = float(cp.max(cp.abs(actual - reference)).item())
    timed_batches = batches * args.cycles
    repeats = 0 if args.verify_only else args.repeats
    for repeat in range(repeats):
        offset = repeat % len(conditions)
        for slots, mode in conditions[offset:] + conditions[:offset]:
            metrics, _ = run_batches(timed_batches, geometry, slots, mode == "sync")
            trials[f"{slots}-slot-{mode}"].append(metrics)

    gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {
        "gpu": gpu["name"].decode(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cupy": cp.__version__,
        "mach": version("mach-beamform"),
        "prepared_iq_sha256": hashlib.sha256(b"".join(batch.tobytes() for batch in batches)).hexdigest(),
        "batch_size": args.batch_size,
        "batches_per_trial": len(timed_batches),
        "max_abs_iq_error": checks,
        "results": {
            condition: {
                "median_completed_frames_per_s": statistics.median(t["completed_frames_per_s"] for t in runs),
                "median_copy_start_to_power_done_p95_ms": statistics.median(
                    t["copy_start_to_power_done_p95_ms"] for t in runs
                ),
                "trials": runs,
            }
            for condition, runs in trials.items()
            if runs
        },
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
