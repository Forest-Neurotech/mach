"""Helper functions for reading PyMUST data."""

import hashlib

import numpy as np
import scipy.io
from jaxtyping import Float

from mach.io.utils import cached_download


def download_pymust_doppler_data() -> dict:
    """Download and load the PyMUST PWI_disk.mat test data.

    This data contains ultrasound RF data from a rotating disk phantom,
    commonly used for benchmarking beamforming algorithms.

    Returns:
        dict: MATLAB data structure containing RF data and parameters
    """
    url = "https://github.com/creatis-ULTIM/PyMUST/raw/fe73cc3911a51b48b36095c2e0ad29c266532447/examples/data/PWI_disk.mat"
    data_file = cached_download(
        url,
        expected_size=2_085_623,
        expected_hash="c349dc1d677c561434fd0e4a74142c4a0a44b7e6ae0a42d446c078a528ef58c1",
        digest=hashlib.sha256,
    )
    mat_data = scipy.io.loadmat(data_file, struct_as_record=False)
    return mat_data


def extract_pymust_params(mat_data: dict) -> dict:
    """Extract PyMUST parameters from loaded MATLAB data.

    Args:
        mat_data: PyMUST data loaded from .mat file
            by `download_pymust_doppler_data`

    Returns:
        dict: PyMUST acquisition parameters including:
            - Nelements: Number of array elements
            - pitch: Element spacing [m]
            - c: Speed of sound [m/s]
            - fs: Sampling frequency [Hz]
            - fc: Center frequency [Hz]
            - t0: Start time [s]
            - fnumber: F-number for beamforming
    """
    mat_param = mat_data["param"][0][0]
    try:
        import pymust.utils

        param = pymust.utils.Param()
    except ImportError:
        # pymust.utils.Param wraps a dict, so if we are not using
        # pymust in the test (and thus don't have pymust.utils),
        # we can simply use a dict instead
        param = {}
    for k, c in mat_param.__dict__.items():
        if k == "_fieldnames":
            continue
        if c.size == 1:
            c = c[0][0]
        param[k] = c
    if "fnumber" not in param:
        # Set a default f-number for other beamformers to use
        param["fnumber"] = 1.0
    return param


def linear_probe_positions(n_elements: int, pitch: float) -> Float[np.ndarray, "{n_elements} xyz=3"]:
    """Generate element positions for a linear array.

    Args:
        n_elements: Number of elements
        pitch: Element spacing [m]

    Returns:
        Element positions [x, y, z] with shape (N_elements, 3)
    """
    # Linear array along x-axis
    x_positions = np.arange(n_elements) * pitch
    x_positions -= np.mean(x_positions)  # Center at origin
    y_positions = np.zeros(n_elements)
    z_positions = np.zeros(n_elements)

    positions = np.column_stack([x_positions, y_positions, z_positions]).astype(np.float32)
    return positions


def scan_grid(*axes: np.ndarray) -> Float[np.ndarray, "points {len(axes)}"]:
    """Return a flattened meshgrid of the given axes.

    Example
        >>> x = np.linspace(-1.25e-2, 1.25e-2, num=251, endpoint=True)
        >>> y = np.array([0.0])
        >>> z = np.linspace(1e-2, 3.5e-2, num=251, endpoint=True)
        >>> grid = scan_grid(x, y, z)
        >>> grid.shape
        (63001, 3)
        >>> grid
    """
    num_xyz = len(axes)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, num_xyz)
