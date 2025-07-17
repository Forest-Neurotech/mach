"""Tests for geometry coordinate conversion functions."""

import math

import array_api_strict as xp
import pytest
from beartype import beartype
from jaxtyping import jaxtyped

from mach.geometry import spherical_to_cartesian as spherical_unchecked
from mach.geometry import ultrasound_angles_to_cartesian as az_el_unchecked

# type-check functions during unit tests
ultrasound_angles_to_cartesian = jaxtyped(typechecker=beartype)(az_el_unchecked)
spherical_to_cartesian = jaxtyped(typechecker=beartype)(spherical_unchecked)


@pytest.mark.no_cuda
class TestAzElToCartesian:
    """Test suite for ultrasound azimuth/elevation to Cartesian conversion."""

    def test_scalar_zero_angles(self):
        """Test scalar conversion with zero angles."""
        x, y, z = ultrasound_angles_to_cartesian(0.0, 0.0, radius_m=1.0)

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(1.0)

    def test_scalar_azimuth_only(self):
        """Test scalar conversion with azimuth rotation only."""
        # 90 degrees azimuth should rotate from z-axis to x-axis
        x, y, z = ultrasound_angles_to_cartesian(math.pi / 2, 0.0, radius_m=2.0)

        assert x == pytest.approx(2.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(0.0)

    def test_scalar_elevation_only(self):
        """Test scalar conversion with elevation rotation only."""
        # 90 degrees elevation should rotate from z-axis to y-axis
        x, y, z = ultrasound_angles_to_cartesian(0.0, math.pi / 2, radius_m=1.5)

        assert x == pytest.approx(0.0)
        assert y == pytest.approx(1.5)
        assert z == pytest.approx(0.0)

    def test_scalar_combined_angles(self):
        """Test scalar conversion with both azimuth and elevation."""
        # 45 degrees azimuth, 45 degrees elevation
        azimuth = math.pi / 4
        elevation = math.pi / 4
        radius = 1.0

        x, y, z = ultrasound_angles_to_cartesian(azimuth, elevation, radius_m=radius)

        # Expected values based on the ultrasound convention:
        # x = r * sin(azimuth)
        # y = r * sin(elevation) * cos(azimuth)
        # z = r * cos(elevation) * cos(azimuth)
        expected_x = radius * math.sin(azimuth)
        expected_y = radius * math.sin(elevation) * math.cos(azimuth)
        expected_z = radius * math.cos(elevation) * math.cos(azimuth)

        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(expected_y)
        assert z == pytest.approx(expected_z)

    def test_scalar_negative_angles(self):
        """Test scalar conversion with negative angles."""
        x, y, z = ultrasound_angles_to_cartesian(-math.pi / 4, -math.pi / 6, radius_m=2.0)

        # Verify the conversion follows the expected formula
        azimuth = -math.pi / 4
        elevation = -math.pi / 6
        radius = 2.0

        expected_x = radius * math.sin(azimuth)
        expected_y = radius * math.sin(elevation) * math.cos(azimuth)
        expected_z = radius * math.cos(elevation) * math.cos(azimuth)

        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(expected_y)
        assert z == pytest.approx(expected_z)

    def test_scalar_norm_validation_pass(self):
        """Test scalar conversion with norm validation enabled (should pass)."""
        x, y, z = ultrasound_angles_to_cartesian(math.pi / 6, math.pi / 4, radius_m=3.0)

        # Calculate actual norm
        norm = math.sqrt(x * x + y * y + z * z)
        assert norm == pytest.approx(3.0, abs=1e-6)

    def test_scalar_norm_validation_disabled(self):
        """Test scalar conversion with norm validation disabled."""
        # This should work without raising any errors
        x, y, z = ultrasound_angles_to_cartesian(math.pi / 3, math.pi / 3, radius_m=1.0)

        # Still verify the calculation is correct
        norm = math.sqrt(x * x + y * y + z * z)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_array_single_point(self):
        """Test array conversion with a single point."""
        azimuth = xp.asarray([math.pi / 4])
        elevation = xp.asarray([math.pi / 3])
        radius = xp.asarray([2.0])

        result = ultrasound_angles_to_cartesian(azimuth, elevation, radius_m=radius)

        assert result.shape == (1, 3)
        # Verify against expected calculation
        expected_x = 2.0 * math.sin(math.pi / 4)
        expected_y = 2.0 * math.sin(math.pi / 3) * math.cos(math.pi / 4)
        expected_z = 2.0 * math.cos(math.pi / 3) * math.cos(math.pi / 4)

        assert float(result[0, 0]) == pytest.approx(expected_x)
        assert float(result[0, 1]) == pytest.approx(expected_y)
        assert float(result[0, 2]) == pytest.approx(expected_z)

    def test_array_multiple_points(self):
        """Test array conversion with multiple points."""
        azimuth = xp.asarray([0.0, math.pi / 2, -math.pi / 4])
        elevation = xp.asarray([0.0, 0.0, math.pi / 6])
        radius = xp.asarray([1.0, 2.0, 1.5])

        result = ultrasound_angles_to_cartesian(azimuth, elevation, radius_m=radius)

        assert result.shape == (3, 3)

        # Check first point (0, 0) -> (0, 0, 1)
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[0, 1]) == pytest.approx(0.0)
        assert float(result[0, 2]) == pytest.approx(1.0)

        # Check second point (π/2, 0) -> (2, 0, 0)
        assert float(result[1, 0]) == pytest.approx(2.0)
        assert float(result[1, 1]) == pytest.approx(0.0)
        assert float(result[1, 2]) == pytest.approx(0.0)

    def test_array_norm_validation_pass(self):
        """Test array conversion with norm validation (should pass)."""
        azimuth = xp.asarray([0.0, math.pi / 4])
        elevation = xp.asarray([0.0, math.pi / 6])
        radius = xp.asarray([1.0, 2.0])

        result = ultrasound_angles_to_cartesian(azimuth, elevation, radius_m=radius)

        # Check norms
        norms = xp.sqrt(xp.sum(result * result, axis=-1))
        assert float(norms[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(norms[1]) == pytest.approx(2.0, abs=1e-6)

    def test_mixed_scalar_array_inputs(self):
        """Test with mixed scalar and array inputs."""
        azimuth = xp.asarray([0.0, math.pi / 2])
        elevation = 0.0  # scalar
        radius = 1.0  # scalar

        result = ultrasound_angles_to_cartesian(azimuth, elevation, radius_m=radius)

        assert result.shape == (2, 3)
        assert float(result[0, 2]) == pytest.approx(1.0)  # First point z-coordinate
        assert float(result[1, 0]) == pytest.approx(1.0)  # Second point x-coordinate


@pytest.mark.no_cuda
class TestSphericalToCartesian:
    """Test suite for spherical to Cartesian coordinate conversion."""

    def test_scalar_zero_angles(self):
        """Test scalar conversion with zero angles."""
        x, y, z = spherical_to_cartesian(0.0, 0.0, radius_m=1.0)

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(1.0)

    def test_scalar_known_angles(self):
        """Test scalar conversion with known angle combinations."""
        # Use smaller angles that don't violate the π/2 constraint
        # θ=π/3, φ=0
        x, y, z = spherical_to_cartesian(math.pi / 3, 0.0, radius_m=1.0)
        expected_x = math.sin(math.pi / 3) * math.cos(0.0)
        expected_z = math.cos(math.pi / 3)
        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(0.0)
        assert z == pytest.approx(expected_z)

        # θ=π/4, φ=π/4
        x, y, z = spherical_to_cartesian(math.pi / 4, math.pi / 4, radius_m=1.0)
        expected_x = math.sin(math.pi / 4) * math.cos(math.pi / 4)
        expected_y = math.sin(math.pi / 4) * math.sin(math.pi / 4)
        expected_z = math.cos(math.pi / 4)

        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(expected_y)
        assert z == pytest.approx(expected_z)

    def test_scalar_45_degree_angles(self):
        """Test scalar conversion with 45-degree angles."""
        # Use 45-degree angles for cleaner test values
        theta = math.pi / 4
        phi = math.pi / 4
        radius = 1.0

        x, y, z = spherical_to_cartesian(theta, phi, radius_m=radius)

        # Expected values based on spherical formula
        expected_x = radius * math.sin(theta) * math.cos(phi)
        expected_y = radius * math.sin(theta) * math.sin(phi)
        expected_z = radius * math.cos(theta)

        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(expected_y)
        assert z == pytest.approx(expected_z)

    def test_scalar_norm_validation_pass(self):
        """Test scalar conversion with norm validation enabled (should pass)."""
        x, y, z = spherical_to_cartesian(math.pi / 6, math.pi / 4, radius_m=3.0)

        # Calculate actual norm
        norm = math.sqrt(x * x + y * y + z * z)
        assert norm == pytest.approx(3.0, abs=1e-6)

    def test_scalar_norm_validation_disabled(self):
        """Test scalar conversion with norm validation disabled."""
        x, y, z = spherical_to_cartesian(math.pi / 3, math.pi / 6, radius_m=1.0)

        # Still verify the calculation is correct
        norm = math.sqrt(x * x + y * y + z * z)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_array_single_point(self):
        """Test array conversion with a single point."""
        theta = xp.asarray([math.pi / 6])
        phi = xp.asarray([math.pi / 4])

        result = spherical_to_cartesian(theta, phi, radius_m=2.0)

        assert result.shape == (1, 3)

    def test_array_multiple_points(self):
        """Test array conversion with multiple points."""
        theta = xp.asarray([0.0, math.pi / 4, math.pi / 3])
        phi = xp.asarray([0.0, math.pi / 4, math.pi / 6])
        radius = xp.asarray([1.0, 2.0, 1.5])

        result = spherical_to_cartesian(theta, phi, radius_m=radius)

        assert result.shape == (3, 3)

        # Check first point (0, 0) -> (0, 0, 1)
        assert float(result[0, 0]) == pytest.approx(0.0)
        assert float(result[0, 1]) == pytest.approx(0.0)
        assert float(result[0, 2]) == pytest.approx(1.0)

    def test_array_norm_validation_pass(self):
        """Test array norm validation (should pass)."""
        theta = xp.asarray([math.pi / 6, math.pi / 4])
        phi = xp.asarray([0.0, math.pi / 6])
        radius = xp.asarray([1.0, 2.0])

        result = spherical_to_cartesian(theta, phi, radius_m=radius)

        # Check norms
        norms = xp.sqrt(xp.sum(result * result, axis=-1))
        assert float(norms[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(norms[1]) == pytest.approx(2.0, abs=1e-6)

    def test_mixed_scalar_array_inputs(self):
        """Test with mixed scalar and array inputs."""
        theta = xp.asarray([0.0, math.pi / 4])
        phi = 0.0  # scalar

        result = spherical_to_cartesian(theta, phi, radius_m=1.0)

        assert result.shape == (2, 3)
        assert float(result[0, 2]) == pytest.approx(1.0)  # First point z-coordinate


@pytest.mark.no_cuda
class TestGeometryEdgeCases:
    """Test edge cases and boundary conditions for both functions."""

    def test_very_small_angles(self):
        """Test with very small angles close to zero."""
        small_angle = 1e-10

        # Test ultrasound_angles_to_cartesian
        x, y, z = ultrasound_angles_to_cartesian(small_angle, small_angle, radius_m=1.0)
        assert z == pytest.approx(1.0, abs=1e-6)

        # Test spherical_to_cartesian
        x, y, z = spherical_to_cartesian(small_angle, small_angle, radius_m=1.0)
        assert z == pytest.approx(1.0, abs=1e-6)

    def test_integer_inputs(self):
        """Test that integer inputs work correctly."""
        # Test ultrasound_angles_to_cartesian with integers
        x, y, z = ultrasound_angles_to_cartesian(0, 0, radius_m=1)
        assert isinstance(x, float)
        assert z == pytest.approx(1.0)

        # Test spherical_to_cartesian with integers
        x, y, z = spherical_to_cartesian(0, 0, radius_m=2)
        assert isinstance(x, float)
        assert z == pytest.approx(2.0)

    def test_different_radius_values(self):
        """Test with various radius values."""
        for radius in [0.1, 1.0, 10.0, 100.0]:
            x, y, z = ultrasound_angles_to_cartesian(0.0, 0.0, radius_m=radius)
            assert z == pytest.approx(radius)

            x, y, z = spherical_to_cartesian(0.0, 0.0, radius_m=radius)
            assert z == pytest.approx(radius)

    def test_coordinate_system_consistency(self):
        """Test that both functions produce geometrically consistent results."""
        # Test a few known points where we can verify the geometry

        # Point along positive x-axis - use smaller angle
        x1, y1, z1 = ultrasound_angles_to_cartesian(math.pi / 2, 0.0, radius_m=1.0)
        x2, y2, z2 = spherical_to_cartesian(math.pi / 3, 0.0, radius_m=1.0)  # Use different but valid angle

        # Verify that our coordinate transformations make geometric sense
        # For ultrasound angles: 90° azimuth should put point on x-axis
        assert x1 == pytest.approx(1.0, abs=1e-6)
        assert y1 == pytest.approx(0.0, abs=1e-6)
        assert z1 == pytest.approx(0.0, abs=1e-6)

        # For spherical: verify the norm is preserved
        norm2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        assert norm2 == pytest.approx(1.0, abs=1e-6)
