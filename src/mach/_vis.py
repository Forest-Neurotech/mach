"""Visualization utilities for test-diagnostics."""

from copy import copy
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def db(x):
    """Convert to dB scale."""
    x = np.where(x == 0, np.finfo(float).eps, x)  # Avoid log of zero
    return 20 * np.log10(np.abs(x))


def db_zero(data):
    """Convert a matrix to dB scale with max at zero."""
    data_db = db(data)  # Convert to dB scale with max at zero
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
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # Skip if matplotlib not available

    output_dir.mkdir(parents=True, exist_ok=True)

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

    if reference_result is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        max_value = max(np.max(np.abs(our_img)), np.max(np.abs(ref_img)))

        # Our result - convert to dB
        our_img_db = db(our_img / max_value)
        im1 = axes[0, 0].imshow(
            our_img_db.T, aspect="auto", origin="upper", cmap="hot", vmin=-40, vmax=0, extent=extent
        )
        axes[0, 0].set_title(our_label)
        axes[0, 0].set_xlabel("Lateral [cm]")
        axes[0, 0].set_ylabel("Depth [cm]")
        cbar1 = plt.colorbar(im1, ax=axes[0, 0])
        cbar1.set_label("dB")

        # Reference result - convert to dB
        ref_img_db = db(ref_img / max_value)
        im2 = axes[0, 1].imshow(
            ref_img_db.T, aspect="auto", origin="upper", cmap="hot", vmin=-40, vmax=0, extent=extent
        )
        axes[0, 1].set_title(reference_label)
        axes[0, 1].set_xlabel("Lateral [cm]")
        axes[0, 1].set_ylabel("Depth [cm]")
        cbar2 = plt.colorbar(im2, ax=axes[0, 1])
        cbar2.set_label("dB")

        # Difference (linear scale)
        diff_img = our_img - ref_img
        vmax = np.max(np.abs(diff_img))
        im3 = axes[1, 0].imshow(
            diff_img.T, aspect="auto", origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent
        )
        axes[1, 0].set_title(f"Difference ({our_label} - {reference_label})")
        axes[1, 0].set_xlabel("Lateral [cm]")
        axes[1, 0].set_ylabel("Depth [cm]")
        cbar3 = plt.colorbar(im3, ax=axes[1, 0])
        cbar3.set_label("Linear")

        # Relative difference
        rel_diff = np.abs(diff_img) / (np.abs(ref_img) + 1e-10)
        cmap = copy(plt.cm.hot)
        cmap.set_over("blue", 1.0)
        im4 = axes[1, 1].imshow(rel_diff.T, aspect="auto", origin="upper", cmap=cmap, extent=extent, vmin=0, vmax=1)
        axes[1, 1].set_title("Relative Difference")
        axes[1, 1].set_xlabel("Lateral [cm]")
        axes[1, 1].set_ylabel("Depth [cm]")
        plt.colorbar(im4, ax=axes[1, 1], extend="max")

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        our_img_db = db_zero(our_img)
        im = ax.imshow(our_img_db.T, aspect="auto", origin="upper", cmap="hot", vmin=-40, vmax=0, extent=extent)
        ax.set_title(f"{our_label} ({reference_label} not available)")
        ax.set_xlabel("Lateral [cm]")
        ax.set_ylabel("Depth [cm]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("dB")

    fig.tight_layout()
    fig.savefig(output_dir / f"{test_name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
