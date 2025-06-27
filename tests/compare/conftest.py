"""Tests to compare this package with different beamformers."""

import numpy as np
import pytest

from mach.io.must import (
    download_pymust_doppler_data,
    extract_pymust_params,
    linear_probe_positions,
)


@pytest.fixture(scope="session")
def pymust_data():
    """Download and load the PyMUST PWI_disk.mat data."""
    return download_pymust_doppler_data()


@pytest.fixture(scope="session")
def pymust_params(pymust_data) -> dict:
    return extract_pymust_params(pymust_data)


@pytest.fixture(scope="session")
def pymust_grid():
    # roughly 100 μm grid spacing
    n_x = 251
    n_z = 251

    x_range = np.linspace(-1.25e-2, 1.25e-2, num=n_x, endpoint=True)
    y_range = np.array([0.0])
    z_range = np.linspace(1e-2, 3.5e-2, num=n_z, endpoint=True)

    return x_range, y_range, z_range


@pytest.fixture(scope="session")
def pymust_element_positions(pymust_params):
    """Generate element positions for linear array."""
    config = pymust_params
    n_elements = config["Nelements"]
    pitch = config["pitch"]

    return linear_probe_positions(n_elements, pitch)
