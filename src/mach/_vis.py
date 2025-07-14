"""Visualization utilities for test-diagnostics."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def db(x, power_mode=False):
    """Convert to dB scale.

    Args:
        x: Input data
        power_mode: If True, use 10*log10 (for power data). If False, use 20*log10 (for amplitude data).
    """
    x = np.where(x == 0, np.finfo(float).eps, x)  # Avoid log of zero
    if power_mode:
        return 10 * np.log10(np.abs(x))
    else:
        return 20 * np.log10(np.abs(x))


def db_zero(data, power_mode=False):
    """Convert a matrix to dB scale with max at zero.

    Args:
        data: Input data
        power_mode: If True, use 10*log10 (for power data). If False, use 20*log10 (for amplitude data).
    """
    data_db = db(data, power_mode=power_mode)  # Convert to dB scale with max at zero
    data_db -= np.max(data_db)
    return data_db


def plot_slice(bm_slice, lats, deps, angle):
    """Plot a slice of beamformed data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        bm_slice,
        extent=[lats[0], lats[-1], deps[-1], deps[0]],
        aspect="equal",
        cmap="gray",
        vmin=-40,
        vmax=0,
        origin="upper",
    )
    fig.colorbar(im, label="dB")
    ax.set_xlabel("Lateral distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Beamformed Image - Angle {angle}")
    ax.set_xlim(lats[0], lats[-1])
    fig.tight_layout()
    return fig


def save_debug_figures(
    our_result: np.ndarray,
    reference_result: Optional[np.ndarray],
    grid_shape: tuple[int, ...],
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    output_dir: Path,
    test_name: str,
    our_label: str = "Our Implementation",
    reference_label: str = "Reference Implementation",
    power_mode: bool = False,
    main_cmap: Optional[str] = None,
    diff_cmap: str = "magma",
) -> None:
    """Save debug figures comparing beamforming results.

    Args:
        our_result: Our beamforming result (magnitude/power)
        reference_result: Reference beamforming result (magnitude/power), optional
        grid_shape: Shape to reshape results for plotting (x, y, z) or (x, z)
        x_axis: X-axis coordinates for extent
        z_axis: Z-axis coordinates for extent
        output_dir: Directory to save figures
        test_name: Name for the output file
        our_label: Label for our implementation
        reference_label: Label for reference implementation
        power_mode: If True, use 10*log10 (for power data). If False, use 20*log10 (for amplitude data).
        main_cmap: Colormap for main images. If None, defaults to 'gray' for amplitude, 'hot' for power.
        diff_cmap: Colormap for difference image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # Skip if matplotlib not available

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default colormaps based on data type
    if main_cmap is None:
        if power_mode:
            # register colorcet colormaps
            import colorcet as cc

            main_cmap = plt.get_cmap("cet_fire")
        else:
            main_cmap = "gray"

    # Handle both 2D and 3D grids - take middle slice for 3D
    if len(grid_shape) == 3:
        # Take 2D slice (assume y=0 slice for 3D data)
        our_img = our_result.reshape(grid_shape)[:, 0, :]  # Shape: (x, z)
        if reference_result is not None:
            ref_img = reference_result.reshape(grid_shape)[:, 0, :]
    elif len(grid_shape) == 2:
        # Already 2D
        our_img = our_result.reshape(grid_shape)
        if reference_result is not None:
            ref_img = reference_result.reshape(grid_shape)
    else:
        raise ValueError(f"Unsupported grid shape: {grid_shape}")

    # Convert coordinates to centimeters for axis labels
    x_extent_cm = [x_axis.min() * 100, x_axis.max() * 100]
    z_extent_cm = [z_axis.min() * 100, z_axis.max() * 100]
    extent = [x_extent_cm[0], x_extent_cm[1], z_extent_cm[1], z_extent_cm[0]]

    # Adjust dynamic range based on power_mode
    vmin_main = -20 if power_mode else -40
    vmin_diff = -70 if power_mode else -140
    units = "power [dB]" if power_mode else "amplitude [dB]"

    if reference_result is not None:
        # Create 1x3 layout: ours, reference, difference in dB
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

        max_value = max(np.max(np.abs(our_img)), np.max(np.abs(ref_img)), 1e-12)

        # Our result - convert to dB
        our_img_db = db(our_img / max_value, power_mode=power_mode)
        im1 = axes[0].imshow(
            our_img_db.T, aspect="equal", origin="upper", cmap=main_cmap, vmin=vmin_main, vmax=0, extent=extent
        )
        axes[0].set_title(our_label)
        axes[0].set_xlabel("Lateral [cm]")
        axes[0].set_ylabel("Depth [cm]")
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label(units)

        # Reference result - convert to dB
        ref_img_db = db(ref_img / max_value, power_mode=power_mode)
        im2 = axes[1].imshow(
            ref_img_db.T, aspect="equal", origin="upper", cmap=main_cmap, vmin=vmin_main, vmax=0, extent=extent
        )
        axes[1].set_title(reference_label)
        axes[1].set_xlabel("Lateral [cm]")
        axes[1].set_ylabel("Depth [cm]")
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label(units)

        # Relative difference in dB
        diff_img = our_img - ref_img
        diff_db = db(diff_img / max_value, power_mode=power_mode)

        # For difference plot, use symmetric scale around zero if using diverging colormap
        if diff_cmap in ["RdBu", "RdBu_r", "seismic", "coolwarm"]:
            # Center the colormap around a reasonable middle value for differences
            vmax_diff = 0
            vmin_diff_centered = vmin_diff
            im3 = axes[2].imshow(
                diff_db.T,
                aspect="equal",
                origin="upper",
                cmap=diff_cmap,
                extent=extent,
                vmin=vmin_diff_centered,
                vmax=vmax_diff,
            )
        else:
            # Use the standard hot colormap scaling
            im3 = axes[2].imshow(
                diff_db.T, aspect="equal", origin="upper", cmap=diff_cmap, extent=extent, vmin=vmin_diff, vmax=0
            )

        axes[2].set_title("Difference (dB, 0dB = max(ref, our))")
        axes[2].set_xlabel("Lateral [cm]")
        axes[2].set_ylabel("Depth [cm]")
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.set_label(units)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        our_img_db = db_zero(our_img, power_mode=power_mode)
        im = ax.imshow(
            our_img_db.T, aspect="equal", origin="upper", cmap=main_cmap, vmin=vmin_main, vmax=0, extent=extent
        )
        ax.set_title(f"{our_label} ({reference_label} not available)")
        ax.set_xlabel("Lateral [cm]")
        ax.set_ylabel("Depth [cm]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(units)

    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
