"""Tests for wavefront propagation functions."""

import math

import array_api_strict as xp
import pytest
from beartype import beartype
from jaxtyping import jaxtyped

from mach.wavefront import plane as plane_unchecked
from mach.wavefront import spherical as spherical_unchecked

# type-check functions during unit tests
plane = jaxtyped(typechecker=beartype)(plane_unchecked)
spherical = jaxtyped(typechecker=beartype)(spherical_unchecked)


@pytest.mark.no_cuda
class TestPlaneWave:
    """Test suite for plane wave transmit function."""

    def test_single_point_along_direction(self):
        """Test with a single point directly along the propagation direction."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])  # Unit vector in x-direction
        distance = 5.0
        points = xp.asarray([distance, 0.0, 0.0])  # Point 5 units away in x-direction

        result = plane(origin, points, direction)

        assert float(result) == pytest.approx(distance)

    def test_single_point_45_degrees(self):
        """Test with a point at 45 degrees to the direction."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])  # Unit vector in x-direction
        # Point at 45 degrees: distance sqrt(2), projection should be 1.0
        points = xp.asarray([1.0, 1.0, 0.0])

        result = plane(origin, points, direction)

        # Projection onto x-axis should be 1.0
        assert float(result) == pytest.approx(1.0)

    def test_single_point_perpendicular(self):
        """Test with a point perpendicular to the direction."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([0.0, 5.0, 0.0])  # Perpendicular to x-direction

        result = plane(origin, points, direction)

        # Projection should be 0.0
        assert float(result) == pytest.approx(0.0)

    def test_batch_points_1x3_shape(self):
        """Test with points shape [1, 3]."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([[2.0, 0.0, 0.0]])  # Shape [1, 3]

        result = plane(origin, points, direction)

        assert result.shape == (1,)
        assert float(result[0]) == pytest.approx(2.0)

    def test_batch_points_nx3_shape(self):
        """Test with points shape [N, 3]."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([
            [1.0, 0.0, 0.0],  # Distance 1.0
            [2.0, 1.0, 0.0],  # Distance 2.0 (projection onto x)
            [0.0, 3.0, 0.0],  # Distance 0.0 (perpendicular)
            [-1.0, 0.0, 0.0],  # Distance -1.0 (behind)
        ])

        result = plane(origin, points, direction)

        assert result.shape == (4,)
        expected = [1.0, 2.0, 0.0, -1.0]
        for i, exp in enumerate(expected):
            assert float(result[i]) == pytest.approx(exp)

    def test_non_aligned_direction(self):
        """Test with non-axis-aligned direction vector."""
        origin = xp.asarray([1.0, 1.0, 0.0])
        # Normalized direction vector at 45 degrees in xy-plane
        direction = xp.asarray([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0])

        # Points along the 45-degree line
        points = xp.asarray([
            [1.0, 1.0, 0.0],  # At origin: distance 0
            [2.0, 2.0, 0.0],  # 1 unit along direction: distance sqrt(2)
            [0.0, 0.0, 0.0],  # 1 unit behind: distance -sqrt(2)
        ])

        result = plane(origin, points, direction)

        assert result.shape == (3,)
        expected = [0.0, math.sqrt(2), -math.sqrt(2)]
        for i, exp in enumerate(expected):
            assert float(result[i]) == pytest.approx(exp)

    def test_non_unit_direction_raises_error(self):
        """Test that non-unit direction vectors raise an error."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([2.0, 0.0, 0.0])  # Not a unit vector
        points = xp.asarray([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="direction must be a unit vector"):
            plane(origin, points, direction)

    def test_different_origins(self):
        """Test with non-zero origin."""
        origin = xp.asarray([2.0, 3.0, 1.0])
        direction = xp.asarray([0.0, 1.0, 0.0])  # y-direction
        points = xp.asarray([2.0, 8.0, 1.0])  # 5 units in y from origin

        result = plane(origin, points, direction)

        assert float(result) == pytest.approx(5.0)

    def test_3d_direction_vector(self):
        """Test with a 3D direction vector."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        # Normalized direction vector in 3D
        direction = xp.asarray([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])

        # Point along the 3D diagonal
        points = xp.asarray([math.sqrt(3), math.sqrt(3), math.sqrt(3)])

        result = plane(origin, points, direction)

        # Dot product should give distance 3.0
        assert float(result) == pytest.approx(3.0)

    def test_zero_distance_points(self):
        """Test with points at the origin."""
        origin = xp.asarray([5.0, 5.0, 5.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([5.0, 5.0, 5.0])  # Same as origin

        result = plane(origin, points, direction)

        assert float(result) == pytest.approx(0.0)


@pytest.mark.no_cuda
class TestSphericalWave:
    """Test suite for spherical wave transmit function."""

    def test_focus_at_origin_point_on_axis(self):
        """Test focused wave with focus at origin and point on z-axis."""
        origin = xp.asarray([0.0, 0.0, -5.0])  # Transducer 5 units behind focus
        focus = xp.asarray([0.0, 0.0, 0.0])  # Focus at origin
        points = xp.asarray([0.0, 0.0, 3.0])  # Point 3 units ahead of focus

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 3.0
        # origin_sign = +1 (focus ahead of origin), point_sign = -1 (focus behind point)
        # result = 5.0 * 1 - 3.0 * (-1) = 5.0 + 3.0 = 8.0
        assert float(result) == pytest.approx(8.0)

    def test_diverging_wave_focus_behind_origin(self):
        """Test diverging wave with focus behind the origin."""
        origin = xp.asarray([0.0, 0.0, 0.0])  # Transducer at origin
        focus = xp.asarray([0.0, 0.0, -5.0])  # Virtual focus behind transducer
        points = xp.asarray([0.0, 0.0, 3.0])  # Point ahead of transducer

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 8.0
        # origin_sign = -1 (focus behind origin), point_sign = -1 (focus behind point)
        # result = 5.0 * (-1) - 8.0 * (-1) = -5.0 + 8.0 = 3.0
        assert float(result) == pytest.approx(3.0)

    def test_point_at_focus(self):
        """Test with point located at the focus."""
        origin = xp.asarray([0.0, 0.0, -2.0])
        focus = xp.asarray([0.0, 0.0, 0.0])
        points = xp.asarray([0.0, 0.0, 0.0])  # Point at focus

        result = spherical(origin, points, focus)

        # origin_focus_dist = 2.0, focus_point_dist = 0.0
        # origin_sign = +1, point_sign undefined (but multiplied by 0)
        # result = 2.0 * 1 - 0.0 * ? = 2.0
        assert float(result) == pytest.approx(2.0)

    def test_point_at_origin(self):
        """Test with point located at the origin (transducer position)."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 5.0])
        points = xp.asarray([0.0, 0.0, 0.0])  # Point at origin

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 5.0
        # origin_sign = +1 (focus ahead), point_sign = +1 (focus ahead of point)
        # result = 5.0 * 1 - 5.0 * 1 = 0.0
        assert float(result) == pytest.approx(0.0)

    def test_batch_points_nx3_shape(self):
        """Test with multiple points in batch."""
        origin = xp.asarray([0.0, 0.0, -3.0])
        focus = xp.asarray([0.0, 0.0, 0.0])
        points = xp.asarray([
            [0.0, 0.0, 0.0],  # At focus
            [0.0, 0.0, 2.0],  # 2 units ahead of focus
            [0.0, 0.0, -1.0],  # 1 unit behind focus
            [0.0, 2.0, 0.0],  # 2 units to side of focus
        ])

        result = spherical(origin, points, focus)

        assert result.shape == (4,)

        # Expected calculations:
        # Point 0: origin_focus=3.0, focus_point=0.0, signs=(+1,?), result=3.0
        # Point 1: origin_focus=3.0, focus_point=2.0, signs=(+1,-1), result=3.0+2.0=5.0
        # Point 2: origin_focus=3.0, focus_point=1.0, signs=(+1,+1), result=3.0-1.0=2.0
        # Point 3: origin_focus=3.0, focus_point=2.0, signs=(+1,?), result=3.0-0.0=3.0
        expected = [3.0, 5.0, 2.0, 3.0]
        for i, exp in enumerate(expected):
            assert float(result[i]) == pytest.approx(exp)

    def test_off_axis_geometry(self):
        """Test with off-axis transducer and focus positions."""
        origin = xp.asarray([2.0, 0.0, -1.0])
        focus = xp.asarray([0.0, 0.0, 2.0])
        points = xp.asarray([1.0, 0.0, 4.0])

        result = spherical(origin, points, focus)

        # origin_focus_dist = sqrt(4 + 0 + 9) = sqrt(13)
        # focus_point_dist = sqrt(1 + 0 + 4) = sqrt(5)
        # origin_sign = +1 (focus.z > origin.z: 2 > -1)
        # point_sign = -1 (focus.z < point.z: 2 < 4)
        expected = math.sqrt(13) * 1 - math.sqrt(5) * (-1)
        expected = math.sqrt(13) + math.sqrt(5)

        assert float(result) == pytest.approx(expected)

    def test_3d_displacement(self):
        """Test with full 3D displacement vectors."""
        origin = xp.asarray([1.0, 1.0, 1.0])
        focus = xp.asarray([2.0, 3.0, 4.0])
        points = xp.asarray([3.0, 2.0, 5.0])

        result = spherical(origin, points, focus)

        # origin_focus_dist = sqrt((2-1)^2 + (3-1)^2 + (4-1)^2) = sqrt(1+4+9) = sqrt(14)
        # focus_point_dist = sqrt((3-2)^2 + (2-3)^2 + (5-4)^2) = sqrt(1+1+1) = sqrt(3)
        # origin_sign = +1 (focus.z > origin.z: 4 > 1)
        # point_sign = -1 (focus.z < point.z: 4 < 5)
        expected = math.sqrt(14) * 1 - math.sqrt(3) * (-1)
        expected = math.sqrt(14) + math.sqrt(3)

        assert float(result) == pytest.approx(expected)

    def test_symmetric_geometry(self):
        """Test with symmetric geometry around focus."""
        origin = xp.asarray([0.0, 0.0, -5.0])
        focus = xp.asarray([0.0, 0.0, 0.0])
        points = xp.asarray([0.0, 0.0, 5.0])  # Symmetric with origin around focus

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 5.0
        # origin_sign = +1, point_sign = -1
        # result = 5.0 * 1 - 5.0 * (-1) = 10.0
        assert float(result) == pytest.approx(10.0)

    def test_batch_points_1x3_shape(self):
        """Test with points shape [1, 3]."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 3.0])
        points = xp.asarray([[0.0, 0.0, 6.0]])  # Shape [1, 3]

        result = spherical(origin, points, focus)

        assert result.shape == (1,)
        # origin_focus_dist = 3.0, focus_point_dist = 3.0
        # origin_sign = +1, point_sign = -1
        # result = 3.0 * 1 - 3.0 * (-1) = 6.0
        assert float(result[0]) == pytest.approx(6.0)
