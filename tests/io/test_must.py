"""Test loading PyMUST data.

Uses jaxtyping and beartype to check type-hints at runtime.
"""

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped

from mach.io.must import (
    download_pymust_doppler_data,
    extract_pymust_params,
    linear_probe_positions,
    scan_grid,
)


@pytest.fixture
def pymust_doppler_data():
    return download_pymust_doppler_data()


@pytest.mark.no_cuda
def test_download_pymust_doppler_data(pymust_doppler_data):
    """Test downloading PyMUST data."""
    assert pymust_doppler_data is not None
    assert len(pymust_doppler_data) > 0
    assert "param" in pymust_doppler_data
    assert "RF" in pymust_doppler_data


@pytest.mark.no_cuda
def test_extract_pymust_params(pymust_doppler_data):
    """Test extracting PyMUST parameters."""
    params = jaxtyped(extract_pymust_params, typechecker=beartype)(pymust_doppler_data)
    assert params is not None
    assert len(params) > 0
    assert "Nelements" in params
    assert "c" in params


@pytest.mark.no_cuda
def test_linear_probe_positions():
    """Test generating linear probe positions."""
    num_elements = 10
    positions = jaxtyped(linear_probe_positions, typechecker=beartype)(10, 0.1)
    assert isinstance(positions, np.ndarray)
    assert positions.shape == (num_elements, 3)
    np.testing.assert_allclose(positions.mean(axis=0), np.zeros(3), atol=1e-8)


@pytest.mark.no_cuda
def test_scan_grid():
    """Test generating a scan grid."""
    num_x = 3
    num_y = 4
    num_z = 5
    grid = jaxtyped(scan_grid, typechecker=beartype)(
        np.linspace(0, 1, num_x), np.linspace(0, 1, num_y), np.linspace(0, 1, num_z)
    )
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (num_x * num_y * num_z, 3)
