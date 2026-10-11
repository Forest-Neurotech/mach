"""Tests for wavefront propagation functions."""

import math

import array_api_strict as xp
import pytest
from array_api_extra.testing import assert_close
from beartype import beartype
from jaxtyping import jaxtyped

from mach.wavefront import earliest_arrival as earliest_arrival_unchecked
from mach.wavefront import plane as plane_unchecked
from mach.wavefront import spherical as spherical_unchecked

# type-check functions during unit tests
earliest_arrival = jaxtyped(typechecker=beartype)(earliest_arrival_unchecked)
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

        assert_close(result, xp.asarray(distance))

    def test_single_point_45_degrees(self):
        """Test with a point at 45 degrees to the direction."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])  # Unit vector in x-direction
        # Point at 45 degrees: distance sqrt(2), projection should be 1.0
        points = xp.asarray([1.0, 1.0, 0.0])

        result = plane(origin, points, direction)

        # Projection onto x-axis should be 1.0
        assert_close(result, xp.asarray(1.0))

    def test_single_point_perpendicular(self):
        """Test with a point perpendicular to the direction."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([0.0, 5.0, 0.0])  # Perpendicular to x-direction

        result = plane(origin, points, direction)

        # Projection should be 0.0
        assert_close(result, xp.asarray(0.0))

    def test_batch_points(self):
        """Test with multiple points."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([
            [1.0, 0.0, 0.0],  # Distance 1.0
            [2.0, 1.0, 0.0],  # Distance 2.0 (projection onto x)
            [0.0, 3.0, 0.0],  # Distance 0.0 (perpendicular)
            [-1.0, 0.0, 0.0],  # Distance -1.0 (behind)
        ])

        result = plane(origin, points, direction)

        assert_close(result, xp.asarray([1.0, 2.0, 0.0, -1.0]))

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

        assert_close(result, xp.asarray([0.0, math.sqrt(2), -math.sqrt(2)]))

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

        assert_close(result, xp.asarray(5.0))

    def test_3d_direction_vector(self):
        """Test with a 3D direction vector."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        # Normalized direction vector in 3D
        direction = xp.asarray([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])

        # Point along the 3D diagonal
        points = xp.asarray([math.sqrt(3), math.sqrt(3), math.sqrt(3)])

        result = plane(origin, points, direction)

        # Dot product should give distance 3.0
        assert_close(result, xp.asarray(3.0))

    def test_zero_distance_points(self):
        """Test with points at the origin."""
        origin = xp.asarray([5.0, 5.0, 5.0])
        direction = xp.asarray([1.0, 0.0, 0.0])
        points = xp.asarray([5.0, 5.0, 5.0])  # Same as origin

        result = plane(origin, points, direction)

        assert_close(result, xp.asarray(0.0))


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
        assert_close(result, xp.asarray(8.0))

    def test_diverging_wave_focus_behind_origin(self):
        """Test diverging wave with focus behind the origin."""
        origin = xp.asarray([0.0, 0.0, 0.0])  # Transducer at origin
        focus = xp.asarray([0.0, 0.0, -5.0])  # Virtual focus behind transducer
        points = xp.asarray([0.0, 0.0, 3.0])  # Point ahead of transducer

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 8.0
        # origin_sign = -1 (focus behind origin), point_sign = -1 (focus behind point)
        # result = 5.0 * (-1) - 8.0 * (-1) = -5.0 + 8.0 = 3.0
        assert_close(result, xp.asarray(3.0))

    def test_point_at_focus(self):
        """Test with point located at the focus."""
        origin = xp.asarray([0.0, 0.0, -2.0])
        focus = xp.asarray([0.0, 0.0, 0.0])
        points = xp.asarray([0.0, 0.0, 0.0])  # Point at focus

        result = spherical(origin, points, focus)

        # origin_focus_dist = 2.0, focus_point_dist = 0.0
        # origin_sign = +1, point_sign undefined (but multiplied by 0)
        # result = 2.0 * 1 - 0.0 * ? = 2.0
        assert_close(result, xp.asarray(2.0))

    def test_point_at_origin(self):
        """Test with point located at the origin (transducer position)."""
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 5.0])
        points = xp.asarray([0.0, 0.0, 0.0])  # Point at origin

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 5.0
        # origin_sign = +1 (focus ahead), point_sign = +1 (focus ahead of point)
        # result = 5.0 * 1 - 5.0 * 1 = 0.0
        assert_close(result, xp.asarray(0.0))

    def test_batch_points(self):
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

        # Expected calculations:
        # Point 0: origin_focus=3.0, focus_point=0.0, signs=(+1,?), result=3.0
        # Point 1: origin_focus=3.0, focus_point=2.0, signs=(+1,-1), result=3.0+2.0=5.0
        # Point 2: origin_focus=3.0, focus_point=1.0, signs=(+1,+1), result=3.0-1.0=2.0
        # Point 3: origin_focus=3.0, focus_point=2.0, signs=(+1,?), result=3.0-0.0=3.0
        assert_close(result, xp.asarray([3.0, 5.0, 2.0, 3.0]))

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

        assert_close(result, xp.asarray(expected))

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

        assert_close(result, xp.asarray(expected))

    def test_symmetric_geometry(self):
        """Test with symmetric geometry around focus."""
        origin = xp.asarray([0.0, 0.0, -5.0])
        focus = xp.asarray([0.0, 0.0, 0.0])
        points = xp.asarray([0.0, 0.0, 5.0])  # Symmetric with origin around focus

        result = spherical(origin, points, focus)

        # origin_focus_dist = 5.0, focus_point_dist = 5.0
        # origin_sign = +1, point_sign = -1
        # result = 5.0 * 1 - 5.0 * (-1) = 10.0
        assert_close(result, xp.asarray(10.0))


@pytest.mark.no_cuda
class TestBoundedSphericalWave:
    """Tests for the rectangular-aperture unified spherical model."""

    @staticmethod
    def aperture():
        return xp.asarray([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])

    def test_matches_spherical_inside_focus_pyramids(self):
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 2.0])
        points = xp.asarray([
            [0.25, 0.25, 1.0],
            [0.25, 0.25, 3.0],
        ])

        bounded = spherical(origin, points, focus, self.aperture())
        legacy = spherical(origin, points, focus)

        assert_close(bounded, legacy)

    def test_interpolates_continuously_across_focal_plane(self):
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 2.0])
        points = xp.asarray([
            [0.75, 0.0, 2.0 - 1e-6],
            [0.75, 0.0, 2.0],
            [0.75, 0.0, 2.0 + 1e-6],
        ])

        result = spherical(origin, points, focus, self.aperture())

        # Nearby points differ physically; tolerance checks continuity, not equality.
        assert_close(
            result,
            xp.asarray([2.0, 2.0, 2.0]),
            atol=2e-6,
            rtol=0.0,
        )

    def test_is_continuous_at_focus_pyramid_boundary(self):
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 2.0])
        points = xp.asarray([
            [0.5, 0.0, 1.0 - 1e-6],
            [0.5, 0.0, 1.0],
            [0.5, 0.0, 1.0 + 1e-6],
        ])

        result = spherical(origin, points, focus, self.aperture())

        # Nearby points differ physically; tolerance checks continuity, not equality.
        assert_close(
            result,
            result * 0.0 + result[1],
            atol=2e-6,
            rtol=0.0,
        )

    def test_uses_both_aperture_dimensions(self):
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, 2.0])
        points = xp.asarray([
            [0.0, 0.75, 2.0],
            [0.75, 0.0, 2.0],
        ])

        result = spherical(origin, points, focus, self.aperture())

        assert_close(result, xp.asarray([2.0, 2.0]))

    def test_steered_focus_is_continuous(self):
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.5, 0.25, 2.0])
        points = xp.asarray([
            [1.0, 0.25, 2.0 - 1e-6],
            [1.0, 0.25, 2.0],
            [1.0, 0.25, 2.0 + 1e-6],
        ])

        result = spherical(origin, points, focus, self.aperture())

        axis = (0.5, 0.25, 2.0)
        lateral = (0.5, 0.0, 0.0)
        focus_distance = math.sqrt(sum(component**2 for component in axis))
        before_distance = math.sqrt(sum((0.5 * axis[i] - lateral[i]) ** 2 for i in range(3)))
        after_distance = math.sqrt(sum((0.5 * axis[i] + lateral[i]) ** 2 for i in range(3)))
        expected_at_focus_plane = focus_distance + (after_distance - before_distance) / 2
        # Nearby points differ physically; tolerance checks continuity, not equality.
        assert_close(
            result,
            xp.asarray([expected_at_focus_plane] * 3),
            atol=2e-6,
            rtol=0.0,
        )

    def test_diverging_focus_uses_aperture_normal(self):
        origin = xp.asarray([0.0, 0.0, 0.0])
        focus = xp.asarray([0.0, 0.0, -2.0])
        points = xp.asarray([[0.5, 0.0, 1.0]])

        result = spherical(origin, points, focus, self.aperture())

        assert_close(
            result,
            xp.asarray([math.sqrt(0.5**2 + 3.0**2) - 2.0]),
        )

    def test_rejects_unordered_rectangle(self):
        aperture = xp.asarray([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        with pytest.raises(ValueError, match="aperture_bounds_m"):
            spherical(
                xp.asarray([0.0, 0.0, 0.0]),
                xp.asarray([[0.0, 0.0, 1.0]]),
                xp.asarray([0.0, 0.0, 2.0]),
                aperture,
            )


@pytest.mark.no_cuda
class TestEarliestArrival:
    """Tests for earliest arrival from explicit element delays."""

    def test_hand_calculated_arrivals(self):
        elements = xp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        delays_s = xp.asarray([0.5, 0.0])
        points = xp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        result = earliest_arrival(
            elements,
            delays_s,
            points,
            sound_speed_m_s=2.0,
        )

        assert_close(result, xp.asarray([1.0, 1.0, 1.0]))

    def test_preserves_leading_point_shape(self):
        elements = xp.asarray([[0.0, 0.0, 0.0]])
        delays_s = xp.asarray([0.0])
        points = xp.asarray([
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            [[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]],
        ])

        result = earliest_arrival(
            elements,
            delays_s,
            points,
            sound_speed_m_s=2.0,
            point_chunk_size=1,
        )

        assert_close(result, xp.asarray([[1.0, 2.0], [3.0, 4.0]]))

    def test_delay_reference_shift_shifts_all_arrivals(self):
        elements = xp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        delays_s = xp.asarray([0.5, 0.0])
        points = xp.asarray([[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])

        baseline = earliest_arrival(
            elements,
            delays_s,
            points,
            sound_speed_m_s=2.0,
        )
        shifted = earliest_arrival(
            elements,
            delays_s + 0.75,
            points,
            sound_speed_m_s=2.0,
        )

        assert_close(shifted - baseline, xp.asarray([1.5, 1.5]))

    def test_chunked_matches_single_reduction(self):
        elements = xp.asarray([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        delays_s = xp.asarray([0.1, 0.05, 0.0])
        points = xp.asarray([
            [-0.5, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 3.0],
        ])

        chunked = earliest_arrival(
            elements,
            delays_s,
            points,
            sound_speed_m_s=2.0,
            element_chunk_size=1,
            point_chunk_size=1,
        )
        direct = earliest_arrival(
            elements,
            delays_s,
            points,
            sound_speed_m_s=2.0,
            element_chunk_size=3,
            point_chunk_size=3,
        )

        assert_close(chunked, direct)

    def test_rejects_nonpositive_sound_speed(self):
        with pytest.raises(ValueError, match="sound_speed_m_s must be positive"):
            earliest_arrival(
                xp.asarray([[0.0, 0.0, 0.0]]),
                xp.asarray([0.0]),
                xp.asarray([[0.0, 0.0, 1.0]]),
                sound_speed_m_s=0.0,
            )
