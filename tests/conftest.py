import contextlib
import hashlib
from pathlib import Path
from typing import cast

import pytest
from pyuff_ustb import ChannelData, Scan, Uff

from mach.io.utils import cached_download

HAS_NUMPY = False
np = None
with contextlib.suppress(ImportError):
    import numpy as np

    HAS_NUMPY = True

HAS_JAX = False
jax = None
jnp = None
with contextlib.suppress(ImportError):
    import jax.numpy as jnp

    HAS_JAX = True


HAS_CUPY = False
cp = None
with contextlib.suppress(ImportError):
    import cupy as cp

    HAS_CUPY = True


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--save-output",
        action="store_true",
        default=False,
        help="Enable debug mode to save beamformed output",
    )
    parser.addoption(
        "--tile-total-frames",
        type=int,
        default=None,
        help="Number of frames to beamform. If larger than the number of frames in the dataset, the dataset will be tiled.",
    )


def pytest_collection_modifyitems(items):
    """Automatically add CUDA marker to all tests unless explicitly marked with no_cuda."""
    for item in items:
        # Check if the test is explicitly marked with no_cuda
        if "no_cuda" in item.keywords:
            # Skip auto-marking for tests that explicitly don't need CUDA
            continue

        # Check if the test already has a cuda marker
        if "cuda" not in item.keywords:
            # Auto-add cuda marker to all tests by default
            item.add_marker(pytest.mark.cuda)


@pytest.fixture
def beamform_params():
    """Default beamforming parameters for tests."""
    return {
        "f_number": 1.0,
        "tukey_alpha": 0.5,  # Tukey window alpha parameter. 0.0 -> rectangular window.
        "time_decim": 1,
    }


@pytest.fixture
def test_data_dir():
    """Base directory for test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def output_dir(request) -> Path | None:
    """Output directory for test results.

    Returns None if the --save-output flag is not set.
    If set, returns tests/output/
    """
    if request.config.getoption("--save-output"):
        return Path(__file__).parent / "output"
    return None


@pytest.fixture(
    params=[
        pytest.param(np, id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(cp, id="cupy", marks=pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")),
        # We need both JAX and CuPy to test JAX, because the `out` array must be writeable (JAX does not support this)
        pytest.param(
            jnp, id="jax", marks=pytest.mark.skipif(not (HAS_JAX and HAS_CUPY), reason="JAX or CuPy not available")
        ),
    ]
)
def xp(request):
    """Fixture providing different array API backends."""
    return request.param


# Data fixtures


@pytest.fixture(scope="session")
def picmus_phantom_resolution_uff() -> Uff:
    """Download the UFF data of the Picmus phantom resolution UFF file."""
    url = "http://www.ustb.no/datasets/PICMUS_experiment_resolution_distortion.uff"
    uff_path = cached_download(
        url,
        expected_size=145_518_524,
        expected_hash="c93af0781daeebf771e53a42a629d4f311407410166ef2f1d227e9d2a1b8c641",
        digest=hashlib.sha256,
    )
    return Uff(str(uff_path))


@pytest.fixture(scope="session")
def picmus_phantom_resolution_scan(picmus_phantom_resolution_uff: Uff) -> Scan:
    """Get the scan object from the UFF data."""
    scan: Uff = picmus_phantom_resolution_uff.read("/scan")
    return cast(Scan, scan)


@pytest.fixture(scope="session")
def picmus_phantom_resolution_channel_data(picmus_phantom_resolution_uff: Uff) -> ChannelData:
    """Get the channel data object from the UFF data."""
    return picmus_phantom_resolution_uff.read("/channel_data")
