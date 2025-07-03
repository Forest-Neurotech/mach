"""Convenience functions for determining the transmit-arrival distance of a wavefront."""

import warnings

from array_api_compat import array_namespace
from jaxtyping import Real

from mach._array_api import Array


def plane(
    origin_m: Real[Array, "xyz=3"],
    points_m: Real[Array, "*points xyz=3"],
    direction: Real[Array, "xyz=3"],
) -> Real[Array, "*points"]:
    """Plane-wave transmit *distance*.

    Plane-wave transmit is described by its propagation direction and origin-position.

    Args:
        origin_m:
            The origin position of the plane-wave in meters.
        points_m:
            The position of the points to compute the transmit-arrival time for, in meters.
        direction:
            The direction of the plane-wave. Must be a unit vector.
            We use direction vector instead of angles because it is less ambiguous.

    Returns:
        The transmit-wavefront-arrival distance for each point in meters.
        To convert to time, divide by sound speed: distance_m / sound_speed_m_s

    Notes:
        Does not check for negative distances.
    """
    xp = array_namespace(origin_m, points_m, direction)

    # Check that the direction is a unit vector
    if hasattr(xp, "linalg"):
        if not xp.abs(xp.linalg.vector_norm(direction) - 1) < 1e-6:
            raise ValueError("direction must be a unit vector")
    else:
        warnings.warn(f"{xp} does not support linalg.vector_norm, assuming direction is a unit vector", stacklevel=2)

    diff: Real[Array, "*points 3"] = points_m - origin_m

    # dot product of diff point with the direction vector gives the distance along the direction
    distance_along_direction = xp.vecdot(diff, direction, axis=-1)

    return distance_along_direction


def spherical(
    origin_m: Real[Array, "xyz=3"],
    points_m: Real[Array, "*points xyz=3"],
    focus_m: Real[Array, "xyz=3"],
) -> Real[Array, "*points"]:
    """Spherical-wave transmit *distance* (also known as focused or diverging waves).

    Spherical waves propagate like a collapsing sphere focusing onto a point,
    or an expanding sphere diverging from a point.

    Args:
        origin_m:
            xyz-position of the transmitting element/sender in meters.
            distance=0 at the origin.
        points_m:
            xyz-positions of the points to compute the transmit-arrival time for, in meters.
        focus_m:
            xyz-position of the focal point where spherical waves converge, in meters.
            sometimes called the source or apex.
            for a focused wave, the focus is in front of the origin.
            for a diverging wave, the focus is behind the origin.
            Note: 'focus' refers to the convergence point, while 'origin' refers
            to the physical transducer element that transmits the wave.

    Returns:
        The transmit-wavefront-arrival distance for each point in meters, calculated as the difference
        between the origin-to-focus distance and the focus-to-point distance.
        To convert to time, divide by sound speed: distance_m / sound_speed_m_s

    Notes:
        Similar to Equation 5 / Figure 2 from:
            https://www.biomecardio.com/publis/ultrasonics21.pdf
            So you think you can DAS? A viewpoint on delay-and-sum beamforming
            Perrot, Polichetti, Varray, Garcia 2021
        Modifications: assume L=0 (use wave-time through origin), and extend to 3D.

        The sign convention accounts for the direction of wave propagation.
        For typical ultrasound imaging where z increases with depth, negative values
        indicate the wavefront arrives before the reference time, positive values after.
    """
    xp = array_namespace(origin_m, points_m, focus_m)

    if not hasattr(xp, "linalg"):
        raise NotImplementedError(
            f"{xp} does not support linalg.vector_norm, please use a different array library or open a Github issue"
        )

    # Distance from origin (transducer element) to focus (focal point)
    origin_focus_dist = xp.linalg.vector_norm(origin_m - focus_m, axis=-1)

    # Distance from focus (focal point) to each point
    focus_point_dist = xp.linalg.vector_norm(focus_m - points_m, axis=-1)

    # Sign convention based on z-direction (depth) for typical ultrasound coordinate system
    # Positive z typically points into the medium (increasing depth)
    z_idx = 2
    origin_sign = xp.sign(focus_m[z_idx] - origin_m[z_idx])
    point_sign = xp.sign(focus_m[z_idx] - points_m[..., z_idx])

    # The spherical wavefront arrival time difference
    return origin_focus_dist * origin_sign - focus_point_dist * point_sign
