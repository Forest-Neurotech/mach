"""Convenience functions for determining the transmit-arrival distance of a wavefront."""

from typing import NamedTuple

from jaxtyping import Float, Real

from mach._array_api import Array, array_namespace, vector_norm

_DEFAULT_ELEMENT_CHUNK = 1024
_DEFAULT_POINT_CHUNK = 32_768


class _ApertureGeometry(NamedTuple):
    """Coordinate frame and dimensions of a rectangular aperture."""

    center_m: Array
    lateral_direction: Array
    elevation_direction: Array
    forward_direction: Array
    lateral_half_width_m: Array
    elevation_half_width_m: Array


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

    direction_norm = vector_norm(direction)
    if not xp.abs(direction_norm - 1) < 1e-6:
        raise ValueError("direction must be a unit vector")

    diff: Real[Array, "*points 3"] = points_m - origin_m

    # dot product of diff point with the direction vector gives the distance along the direction
    distance_along_direction = xp.vecdot(diff, direction, axis=-1)

    return distance_along_direction


def _spherical_distance(
    origin_m: Real[Array, "xyz=3"],
    points_m: Real[Array, "*points xyz=3"],
    focus_m: Real[Array, "xyz=3"],
    forward_m: Real[Array, "xyz=3"] | None = None,
) -> Real[Array, "*points"]:
    """Calculate the ideal spherical or virtual-source branch.

    Args:
        origin_m:
            Transmit reference position in meters.
        points_m:
            Image-point positions in meters.
        focus_m:
            Focus or virtual-source position in meters.
        forward_m:
            Optional array-forward unit vector. When omitted, positive z is forward.

    Returns:
        Spherical transmit-arrival distances in meters.
    """
    xp = array_namespace(origin_m, points_m, focus_m, forward_m)

    origin_focus_dist = vector_norm(origin_m - focus_m, axis=-1)
    focus_point_dist = vector_norm(focus_m - points_m, axis=-1)

    if forward_m is None:
        origin_side = focus_m[2] - origin_m[2]
        point_side = focus_m[2] - points_m[..., 2]
    else:
        origin_side = xp.vecdot(focus_m - origin_m, forward_m, axis=-1)
        point_side = xp.vecdot(focus_m - points_m, forward_m, axis=-1)

    return origin_focus_dist * xp.sign(origin_side) - focus_point_dist * xp.sign(point_side)


def _rectangular_aperture_geometry(
    aperture_bounds_m: Real[Array, "corners=4 xyz=3"],
) -> _ApertureGeometry:
    """Describe the coordinate frame and dimensions of a rectangular aperture.

    Args:
        aperture_bounds_m:
            Coordinates (in meters) of the four rectangular-aperture corners.
            Corners must be ordered around its perimeter using the right-hand
            rule so their normal points into the imaging region
            (for example: lateral-min elevation-min, lateral-max elevation-min,
            lateral-max elevation-max, lateral-min elevation-max).

    Returns:
        Aperture geometry

    Raises:
        ValueError:
            If the corners do not describe a nondegenerate rectangle.
    """
    xp = array_namespace(aperture_bounds_m)

    if aperture_bounds_m.shape != (4, 3):
        raise ValueError("aperture_bounds_m must have shape (4, 3)")

    first_corner_m = aperture_bounds_m[0, ...]
    lateral_corner_m = aperture_bounds_m[1, ...]
    elevation_corner_m = aperture_bounds_m[3, ...]
    aperture_center_m = xp.mean(aperture_bounds_m, axis=0)

    lateral_edge_m = lateral_corner_m - first_corner_m
    lateral_width_m = vector_norm(lateral_edge_m, axis=-1)
    if float(xp.abs(lateral_width_m)) < 1e-12:
        raise ValueError("aperture_bounds_m corners 0 and 1 must not coincide")
    lateral_direction = lateral_edge_m / lateral_width_m

    elevation_edge_m = elevation_corner_m - first_corner_m
    elevation_width_m = vector_norm(elevation_edge_m, axis=-1)
    if float(xp.abs(elevation_width_m)) < 1e-12:
        raise ValueError("aperture_bounds_m corners 0 and 3 must not coincide")
    elevation_direction = elevation_edge_m / elevation_width_m
    edge_dot_product = xp.vecdot(
        lateral_direction,
        elevation_direction,
        axis=-1,
    )
    if float(xp.abs(edge_dot_product)) > 1e-6:
        raise ValueError("aperture_bounds_m adjacent edges must be perpendicular")

    forward_direction = xp.stack(
        (
            lateral_direction[1] * elevation_direction[2] - lateral_direction[2] * elevation_direction[1],
            lateral_direction[2] * elevation_direction[0] - lateral_direction[0] * elevation_direction[2],
            lateral_direction[0] * elevation_direction[1] - lateral_direction[1] * elevation_direction[0],
        ),
        axis=-1,
    )
    forward_norm = vector_norm(forward_direction, axis=-1)
    if float(xp.abs(forward_norm)) < 1e-12:
        raise ValueError("aperture_bounds_m corners are degenerate (not a rectangle)")
    forward_direction = forward_direction / forward_norm

    expected_opposite_corner_m = lateral_corner_m + elevation_corner_m - first_corner_m
    opposite_corner_error_m = vector_norm(
        aperture_bounds_m[2, ...] - expected_opposite_corner_m,
        axis=-1,
    )
    aperture_scale_m = xp.maximum(
        xp.maximum(lateral_width_m, elevation_width_m),
        xp.asarray(1e-12, dtype=lateral_width_m.dtype),
    )
    if float(opposite_corner_error_m / aperture_scale_m) > 1e-6:
        raise ValueError("aperture_bounds_m must describe a rectangle ordered around its perimeter")
    return _ApertureGeometry(
        center_m=aperture_center_m,
        lateral_direction=lateral_direction,
        elevation_direction=elevation_direction,
        forward_direction=forward_direction,
        lateral_half_width_m=lateral_width_m / 2.0,
        elevation_half_width_m=elevation_width_m / 2.0,
    )


def _unified_spherical_bounded(
    origin_m: Real[Array, "xyz=3"],
    points_m: Real[Array, "*points xyz=3"],
    focus_m: Real[Array, "xyz=3"],
    aperture_bounds_m: Real[Array, "corners=4 xyz=3"],
) -> Real[Array, "*points"]:
    """Interpolate spherical distances outside the valid focus pyramids.

    Args:
        origin_m:
            Transmit reference position in meters.
        points_m:
            Image-point positions in meters.
        focus_m:
            Focus or virtual-source position in meters.
        aperture_bounds_m:
            Four rectangular-aperture corners ordered around its perimeter.

    Returns:
        Spherical distances inside the focus pyramids and continuously
        interpolated distances between them.

    Raises:
        ValueError:
            If the focus lies in the aperture plane.
    """
    xp = array_namespace(origin_m, points_m, focus_m, aperture_bounds_m)
    aperture = _rectangular_aperture_geometry(aperture_bounds_m)
    spherical_distance_m = _spherical_distance(
        origin_m,
        points_m,
        focus_m,
        forward_m=aperture.forward_direction,
    )
    focus_axis_m = focus_m - aperture.center_m
    focus_depth_m = xp.vecdot(
        focus_axis_m,
        aperture.forward_direction,
        axis=-1,
    )
    if float(xp.abs(focus_depth_m)) < 1e-12:
        raise ValueError("focus_m must not lie in the aperture plane")

    # normalized_depth is 0 at the aperture and 1 at the focus.
    normalized_depth = (
        xp.vecdot(
            points_m - aperture.center_m,
            aperture.forward_direction,
            axis=-1,
        )
        / focus_depth_m
    )
    focus_axis_points_m = aperture.center_m + normalized_depth[..., None] * focus_axis_m
    offset_from_focus_axis_m = points_m - focus_axis_points_m

    # Scale lateral and elevation offsets by their aperture half-widths.
    # Inside the focus pyramid, the largest scaled offset is at most |1 - depth|.
    normalized_lateral_offset = (
        xp.abs(
            xp.vecdot(
                offset_from_focus_axis_m,
                aperture.lateral_direction,
                axis=-1,
            )
        )
        / aperture.lateral_half_width_m
    )
    normalized_elevation_offset = (
        xp.abs(
            xp.vecdot(
                offset_from_focus_axis_m,
                aperture.elevation_direction,
                axis=-1,
            )
        )
        / aperture.elevation_half_width_m
    )
    normalized_aperture_offset = xp.maximum(
        normalized_lateral_offset,
        normalized_elevation_offset,
    )
    inside_focus_pyramid = normalized_aperture_offset <= xp.abs(normalized_depth - 1.0)

    # A line through P parallel to the focus axis intersects the pyramid
    # boundaries where normalized depth is 1 ± normalized aperture offset.
    near_boundary_depth = -normalized_aperture_offset + 1.0
    far_boundary_depth = normalized_aperture_offset + 1.0
    near_boundary_points_m = points_m + (near_boundary_depth - normalized_depth)[..., None] * focus_axis_m
    far_boundary_points_m = points_m + (far_boundary_depth - normalized_depth)[..., None] * focus_axis_m
    near_boundary_distance_m = _spherical_distance(
        origin_m,
        near_boundary_points_m,
        focus_m,
        forward_m=aperture.forward_direction,
    )
    far_boundary_distance_m = _spherical_distance(
        origin_m,
        far_boundary_points_m,
        focus_m,
        forward_m=aperture.forward_direction,
    )

    boundary_depth_span = xp.where(
        normalized_aperture_offset > 0,
        normalized_aperture_offset * 2.0,
        normalized_aperture_offset * 0.0 + 1.0,
    )
    far_boundary_weight = (normalized_depth - near_boundary_depth) / boundary_depth_span
    interpolated_distance_m = near_boundary_distance_m + far_boundary_weight * (
        far_boundary_distance_m - near_boundary_distance_m
    )

    return xp.where(
        inside_focus_pyramid,
        spherical_distance_m,
        interpolated_distance_m,
    )


def spherical(
    origin_m: Real[Array, "xyz=3"],
    points_m: Real[Array, "*points xyz=3"],
    focus_m: Real[Array, "xyz=3"],
    aperture_bounds_m: Real[Array, "corners=4 xyz=3"] | None = None,
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
        aperture_bounds_m:
            Optional corners of the active transmit aperture as shape ``(4, 3)`` in meters.
            Corners must form a nondegenerate rectangle, be coplanar, and be ordered around
            its perimeter using the right-hand
            rule so their normal points into the imaging region
            (for example: lateral-min elevation-min, lateral-max elevation-min,
            lateral-max elevation-max, lateral-min elevation-max).
            When `aperture_bounds_m` is provided, `spherical` uses the ideal spherical
            equation inside the rectangular focus pyramids and linearly interpolates
            boundary spherical distances between them, following the
            unified pixel-based beamforming idea of Nguyen and Prager
            (https://doi.org/10.1109/TMI.2015.2456982). This removes the
            focal-plane discontinuity of the ideal spherical equation for off-axis
            pixels. Outside the insonified sector the result is a travel-time
            continuity approximation, not a full wave model.

    Returns:
        The transmit-wavefront-arrival distance for each point in meters.
        To convert to time, divide by sound speed: distance_m / sound_speed_m_s

    Raises:
        ValueError:
            If aperture bounds do not describe an ordered rectangle or if the focus lies
            in the aperture plane.

    Notes:
        Without ``aperture_bounds_m``, this matches
        Equation 5 / Figure 2 from Perrot et al., Ultrasonics 2021,
        https://www.biomecardio.com/publis/ultrasonics21.pdf with ``L=0``,
        extended to 3D.

        The sign convention accounts for the direction of wave propagation.
        For typical ultrasound imaging where z increases with depth, negative values
        indicate the wavefront arrives before the reference time, positive values after.
    """
    if aperture_bounds_m is None:
        return _spherical_distance(origin_m, points_m, focus_m)
    if aperture_bounds_m.shape != (4, 3):
        raise ValueError("aperture_bounds_m must have shape (4, 3)")

    # The spherical equation is valid only inside the rectangular focus pyramids.
    # Outside them, use a 3D extension of Nguyen and Prager's unified interpolation.
    return _unified_spherical_bounded(
        origin_m,
        points_m,
        focus_m,
        aperture_bounds_m,
    )


def earliest_arrival(
    element_positions_m: Real[Array, "n_elements xyz=3"],
    element_delays_s: Real[Array, " n_elements"],
    points_m: Float[Array, "*points xyz=3"],
    *,
    sound_speed_m_s: float,
    element_chunk_size: int = _DEFAULT_ELEMENT_CHUNK,
    point_chunk_size: int = _DEFAULT_POINT_CHUNK,
) -> Real[Array, "*points"]:
    """Earliest transmit arrival *distance* from per-element delays.

    Args:
        element_positions_m:
            Positions of the active transmit elements, in meters.
        element_delays_s:
            Per-element transmit delays, in seconds. Delays must share the same
            time reference as the receive chain (for example, normalized so the
            earliest firing element has delay zero).
        points_m:
            Positions of the points to compute the transmit-arrival time for, in meters.
        sound_speed_m_s:
            Speed of sound, in meters per second.
        element_chunk_size:
            Number of elements processed per reduction chunk. Tune for memory on large apertures.
        point_chunk_size:
            Number of image points processed per chunk. Tune for memory on large grids.

    Returns:
        Minimum path length ``min_i (sound_speed_m_s * element_delays_s[i] + |P - e_i|)``
        in meters.
        Divide by sound speed to obtain arrival time in seconds.

    Raises:
        ValueError:
            If sound speed or a chunk size is nonpositive, no elements are
            provided, or the element positions and delays have incompatible
            shapes.

    Notes:
        This is the earliest arrival, not necessarily the strongest pulse time
        The same minimum-over-elements method is used in delay-based DAS paths
        such as PyMUST ``dasmtx``
        (https://github.com/creatis-ULTIM/PyMUST/blob/c605195cd00a39295f5a5650df63f7b171b6a36d/src/pymust/dasmtx.py#L450-L461)
        and ultraspy transmit mode 0
        (https://gitlab.com/pecarlat/ultraspy/-/raw/26b0bc171b5344a2241897b7e63af92e845aa77c/src/ultraspy/cpu/kernels/numpy_cores/das.py).
    """
    if element_chunk_size < 1:
        raise ValueError("element_chunk_size must be at least 1")
    if point_chunk_size < 1:
        raise ValueError("point_chunk_size must be at least 1")
    if sound_speed_m_s <= 0:
        raise ValueError("sound_speed_m_s must be positive")

    xp = array_namespace(element_positions_m, element_delays_s, points_m)

    n_elements = int(element_positions_m.shape[0])
    if n_elements == 0:
        raise ValueError("element_positions_m must contain at least one element")
    if element_delays_s.shape != (n_elements,):
        raise ValueError("element_delays_s must have shape (n_elements,)")
    element_delay_distances_m = element_delays_s * sound_speed_m_s

    points_shape = points_m.shape[:-1]
    points_flat = xp.reshape(points_m, (-1, 3))
    n_points = int(points_flat.shape[0])

    point_results = []
    for point_start in range(0, n_points, point_chunk_size):
        point_end = min(point_start + point_chunk_size, n_points)
        point_batch = points_flat[point_start:point_end, ...]
        earliest_distances_m = None

        for element_start in range(0, n_elements, element_chunk_size):
            element_end = min(element_start + element_chunk_size, n_elements)
            element_batch_m = element_positions_m[element_start:element_end, ...]
            delay_distance_batch_m = element_delay_distances_m[element_start:element_end]

            point_element_offsets_m = point_batch[:, None, :] - element_batch_m[None, :, :]
            propagation_distances_m = vector_norm(
                point_element_offsets_m,
                axis=-1,
            )
            arrival_distances_m = propagation_distances_m + delay_distance_batch_m[None, :]
            chunk_earliest_distances_m = xp.min(arrival_distances_m, axis=1)
            earliest_distances_m = (
                chunk_earliest_distances_m
                if earliest_distances_m is None
                else xp.minimum(earliest_distances_m, chunk_earliest_distances_m)
            )
        assert earliest_distances_m is not None
        point_results.append(earliest_distances_m)

    result = xp.concat(point_results, axis=0) if point_results else xp.asarray([], dtype=points_flat.dtype)
    return xp.reshape(result, points_shape)
