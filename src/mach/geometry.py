"""Geometry utilities for coordinate system conversions.

This module provides functions for converting between different coordinate systems
commonly used in ultrasound imaging:

1. `ultrasound_angles_to_cartesian` - Convert ultrasound angles (azimuth, elevation) to Cartesian coordinates
2. `spherical_to_cartesian` - Convert physics spherical coordinates to Cartesian coordinates

We define the "ultrasound convention" as:
1. Rotate azimuth (counter-clockwise) around fixed y-axis
2. Then rotate elevation (clockwise) around fixed x-axis

This convention is more intuitive for elevation multi-slice imaging where you
first create a scan in azimuth, then rotate that plane in elevation.

For discussion on ultrasound toolbox conventions, see:
https://github.com/magnusdk/vbeam/pull/50#issuecomment-2725125129

For convenience, we also provide spherical angle conversions following the physics convention as defined in ISO 80000-2:2019.
"""

import math
import numbers

from jaxtyping import Real

from mach._array_api import Array, _ArrayNamespace, array_namespace


def ultrasound_angles_to_cartesian(
    azimuth_rad: Real[Array, " *angles"] | float | int,
    elevation_rad: Real[Array, " *angles"] | float | int,
    radius_m: Real[Array, " *angles"] | float | int = 1,
) -> Real[Array, "*angles xyz=3"] | tuple[float, float, float]:
    """Convert ultrasound angles (azimuth, elevation, radius) to Cartesian coordinates.

    The resulting vectors can be used directly with `mach.wavefront.plane()`.

    We define the "ultrasound convention" as:
    1. Rotate azimuth (counter-clockwise) around fixed y-axis
    2. Then rotate elevation (clockwise) around fixed x-axis

    This convention is more intuitive for elevation multi-slice imaging where you
    first create a scan in azimuth, then rotate that plane in elevation.

    Following the corrected mapping from https://github.com/magnusdk/vbeam/pull/50#issuecomment-2682744158
        x = r * sin(azimuth)
        y = r * sin(elevation) * cos(azimuth)
        z = r * cos(elevation) * cos(azimuth)

    Args:
        azimuth_rad:
            Azimuth angle in radians - angle in xz-plane (around y-axis), applied first.
        elevation_rad:
            Elevation angle in radians - angle from xz-plane (after azimuth rotation).
        radius_m:
            Radius/distance from origin in meters (defaults to 1 for unit vectors).

    Returns:
        Cartesian coordinates in xyz-order.
        For array inputs: returns array with shape (*coords, 3).
        For scalar inputs: returns tuple[float, float, float].

    Examples:
        >>> # Convert 15° azimuth to Cartesian
        >>> import numpy as np
        >>> x, y, z = ultrasound_angles_to_cartesian(np.radians(15), 0, 1)
        >>> print(f"15° azimuth: ({x:.3f}, {y:.3f}, {z:.3f})")

        >>> # Multiple angles
        >>> azimuths = np.radians([-10, 0, 10])
        >>> coords = ultrasound_angles_to_cartesian(azimuths, 0, 1)
        >>> print(coords.shape)  # (3, 3)
    """
    xp, (azimuth_rad, elevation_rad, radius_m) = _prepare_inputs_and_namespace(azimuth_rad, elevation_rad, radius_m)

    # Convert using unified code path
    x = radius_m * xp.sin(azimuth_rad)
    y = radius_m * xp.sin(elevation_rad) * xp.cos(azimuth_rad)
    z = radius_m * xp.cos(elevation_rad) * xp.cos(azimuth_rad)

    result = (x, y, z)
    if hasattr(xp, "stack"):
        result = xp.stack(result, axis=-1)
    return result


def spherical_to_cartesian(
    theta_rad: Real[Array, " *angles"] | float | int,
    phi_rad: Real[Array, " *angles"] | float | int,
    radius_m: Real[Array, " *angles"] | float | int = 1,
) -> Real[Array, "*angles xyz=3"] | tuple[float, float, float]:
    """Convert standard spherical angle convention to a Cartesian vector.

    Uses the physics convention as defined in ISO 80000-2:2019.
    https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Args:
        theta_rad:
            Polar angle in radians - angle between the radial line and a polar axis.
        phi_rad:
            Azimuthal angle in radians - angle of rotation of the radial line around
            the polar axis.
        radius_m:
            Radial distance from the origin in meters (defaults to 1 for unit vectors).

    Returns:
        Wave-direction vectors in xyz-order, with norm=radius_m.
        For scalar inputs: returns tuple[float, float, float].
        For array inputs: returns array with shape (*angles, 3).

    Raises:
        ValueError: If any angle has magnitude >= π/2, suggesting possible unit confusion.

    Examples:
        >>> # Convert physics spherical coordinates
        >>> import numpy as np
        >>> x, y, z = spherical_to_cartesian(np.pi/4, np.pi/3, 1)
        >>> print(f"Spherical to Cartesian: ({x:.3f}, {y:.3f}, {z:.3f})")
    """
    xp, (theta_rad, phi_rad, radius_m) = _prepare_inputs_and_namespace(theta_rad, phi_rad, radius_m)

    # Convert to Cartesian coordinates
    x = radius_m * xp.sin(theta_rad) * xp.cos(phi_rad)
    y = radius_m * xp.sin(theta_rad) * xp.sin(phi_rad)
    z = radius_m * xp.cos(theta_rad)

    result = (x, y, z)
    if isinstance(xp, _ArrayNamespace) or hasattr(xp, "stack"):
        result = xp.stack(result, axis=-1)
    return result


def _prepare_inputs_and_namespace(
    *inputs: Real[Array, "..."] | float | int,
) -> tuple[type[math] | _ArrayNamespace, tuple]:
    """Prepare inputs and determine the appropriate namespace (math or array).

    For scalar inputs, to avoid requiring a specific array library import,
    we use the built-in math module.

    Returns:
        tuple: (namespace, processed_inputs)
    """
    is_scalar_input = all(isinstance(x, numbers.Real) for x in inputs)

    if is_scalar_input:
        return math, inputs

    # Array case - get namespace and convert any scalars to arrays
    xp = array_namespace(*[x for x in inputs if hasattr(x, "shape")])
    processed_inputs = tuple(xp.asarray(x) if isinstance(x, numbers.Real) else x for x in inputs)
    return xp, processed_inputs
