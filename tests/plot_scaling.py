#!/usr/bin/env python3
"""
Plot performance scaling results from pytest-benchmark JSON files.

This script creates a 2x3 subplot grid showing:
- Top row: Runtime scaling (seconds)
- Bottom row: Points-per-second scaling
- Columns: Voxels, Receive-Elements, Frames

Usage:
    python tests/plot_scaling.py [json_file_path] [--output output.png]

If no JSON file path is provided, uses the most recent file from .benchmarks/**/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Constants based on PyMUST rotating disk dataset
PYMUST_DATASET_PARAMS = {
    "n_elements": 128,  # From PyMUST L7-4 linear array
    "n_frames": 32,  # Baseline frame count for rotating disk dataset
    "grid_x_size": 251,  # From baseline grid
    "grid_z_size": 251,  # From baseline grid
    "n_voxels": 251 * 251,  # 2D grid: x_grid * z_grid
    "sound_speed_m_s": 1480,
    "grid_z_max_m": 3.5e-2,
}


def find_latest_benchmark_file() -> Path:
    """Find the most recent benchmark JSON file in .benchmarks/**/"""
    benchmark_dir = Path(".benchmarks")
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Error: Benchmark directory not found: {benchmark_dir}")

    # Find all JSON files recursively
    json_files = list(benchmark_dir.glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"Error: No benchmark JSON files found in {benchmark_dir}")

    # Find most recent file by modification time
    return max(json_files, key=lambda f: f.stat().st_mtime)


def load_benchmark_data(json_path: Path) -> dict:
    """Load benchmark data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def parse_scaling_parameter(param_string: str, test_type: str) -> float:
    """Parse scaling parameter from parameter string."""
    if test_type == "voxels":
        # Extract resolution like "res_1e-4" -> 1e-4
        match = re.search(r"res_([0-9.e-]+)", param_string)
        if match:
            return float(match.group(1))
    elif test_type == "elements":
        # Extract multiplier like "1x_elements" -> 1
        match = re.search(r"([0-9]+)x_elements", param_string)
        if match:
            return int(match.group(1))
    elif test_type == "frames":
        # Handle fractional multipliers like "1/32x_frames"
        fraction_match = re.search(r"(\d+)/(\d+)x_frames", param_string)
        if fraction_match:
            numerator = int(fraction_match.group(1))
            denominator = int(fraction_match.group(2))
            return numerator / denominator
        else:
            # Handle integer multipliers like "4x_frames"
            match = re.search(r"([0-9]+)x_frames", param_string)
            if match:
                return int(match.group(1))

    return None


def calculate_scaling_factors(benchmark_data: dict) -> dict[str, list[dict]]:  # noqa: C901
    """Calculate scaling factors for each test type."""
    scaling_data = {"voxels": [], "elements": [], "frames": []}

    for benchmark in benchmark_data["benchmarks"]:
        group = benchmark.get("group", "")

        if group == "scaling_voxels":
            param_value = parse_scaling_parameter(benchmark["param"], "voxels")
            if param_value is not None:
                # Calculate number of voxels based on resolution
                # From the test: grid extent is fixed, resolution varies
                x_extent = 2.5e-2  # -1.25e-2 to 1.25e-2
                z_extent = 2.5e-2  # 1e-2 to 3.5e-2
                n_x = int(x_extent / param_value) + 1
                n_z = int(z_extent / param_value) + 1
                n_voxels = n_x * n_z

                scaling_data["voxels"].append({
                    "resolution": param_value,
                    "n_voxels": n_voxels,
                    "scaling_factor": n_voxels / PYMUST_DATASET_PARAMS["n_voxels"],
                    "mean_time": benchmark["stats"]["mean"],
                    "stddev_time": benchmark["stats"]["stddev"],
                    "name": benchmark["name"],
                })

        elif group == "scaling_elements":
            param_value = parse_scaling_parameter(benchmark["param"], "elements")
            if param_value is not None:
                n_elements = PYMUST_DATASET_PARAMS["n_elements"] * param_value

                scaling_data["elements"].append({
                    "multiplier": param_value,
                    "n_elements": n_elements,
                    "scaling_factor": param_value,
                    "mean_time": benchmark["stats"]["mean"],
                    "stddev_time": benchmark["stats"]["stddev"],
                    "name": benchmark["name"],
                })

        elif group == "scaling_frames":
            param_value = parse_scaling_parameter(benchmark["param"], "frames")
            if param_value is not None:
                n_frames = PYMUST_DATASET_PARAMS["n_frames"] * param_value

                scaling_data["frames"].append({
                    "multiplier": param_value,
                    "n_frames": n_frames,
                    "scaling_factor": param_value,
                    "mean_time": benchmark["stats"]["mean"],
                    "stddev_time": benchmark["stats"]["stddev"],
                    "name": benchmark["name"],
                })

    # Sort by scaling factor and deduplicate by averaging multiple runs
    for key in scaling_data:
        # Group by scaling factor
        grouped = {}
        for item in scaling_data[key]:
            sf = item["scaling_factor"]
            if sf not in grouped:
                grouped[sf] = []
            grouped[sf].append(item)

        # Average multiple runs for each scaling factor
        deduplicated = []
        for items in grouped.values():
            if len(items) == 1:
                deduplicated.append(items[0])
            else:
                # Average the timing results
                avg_mean = sum(item["mean_time"] for item in items) / len(items)
                avg_stddev = np.sqrt(sum(item["stddev_time"] ** 2 for item in items) / len(items))

                # Use the first item as template and update timing
                avg_item = items[0].copy()
                avg_item["mean_time"] = avg_mean
                avg_item["stddev_time"] = avg_stddev
                avg_item["name"] = f"{items[0]['name']} (avg of {len(items)} runs)"
                deduplicated.append(avg_item)

        # Sort by scaling factor
        deduplicated.sort(key=lambda x: x["scaling_factor"])
        scaling_data[key] = deduplicated

    return scaling_data


def calculate_points_per_second(time_seconds: float, scaling_factor: float, test_type: str) -> float:
    """Calculate effective points per second for the given timing and scaling."""
    if test_type == "voxels":
        # Points = receive-elements × voxels × frames
        total_points = (
            PYMUST_DATASET_PARAMS["n_elements"]
            * PYMUST_DATASET_PARAMS["n_voxels"]
            * scaling_factor
            * PYMUST_DATASET_PARAMS["n_frames"]
        )
    elif test_type == "elements":
        # Points = receive-elements × voxels × frames
        total_points = (
            PYMUST_DATASET_PARAMS["n_elements"]
            * scaling_factor
            * PYMUST_DATASET_PARAMS["n_voxels"]
            * PYMUST_DATASET_PARAMS["n_frames"]
        )
    elif test_type == "frames":
        # Points = receive-elements × voxels × frames
        total_points = (
            PYMUST_DATASET_PARAMS["n_elements"]
            * PYMUST_DATASET_PARAMS["n_voxels"]
            * PYMUST_DATASET_PARAMS["n_frames"]
            * scaling_factor
        )
    else:
        return 0

    return total_points / time_seconds


def create_scaling_plots(  # noqa: C901
    scaling_data: dict[str, list[dict]], benchmark_data: dict, output_path: Optional[str] = None
) -> None:
    """Create 2x3 subplot grid showing scaling performance with shared x-axis."""

    # Set up the plot style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create 2x3 subplot grid with shared x-axis and shared y-axes within each row
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey="row")

    # Test types and their display names
    test_types = [
        ("voxels", "Scaling Voxels", "n_voxels"),
        ("elements", "Scaling Receive-Elements", "n_elements"),
        ("frames", "Scaling Frames", "n_frames"),
    ]

    # Collect all total points values to determine x-axis limits
    all_total_points = []
    secondary_axes = []  # Store references to secondary axes

    for col, (test_type, title, x_param) in enumerate(test_types):
        data = scaling_data[test_type]

        if not data:
            continue

        # Extract data for plotting
        x_values = [d[x_param] for d in data]
        y_times = [d["mean_time"] for d in data]
        y_errors = [d["stddev_time"] for d in data]

        # Calculate total points for main x-axis
        total_points = []
        for d in data:
            if test_type == "voxels":
                points = PYMUST_DATASET_PARAMS["n_elements"] * d["n_voxels"] * PYMUST_DATASET_PARAMS["n_frames"]
            elif test_type == "elements":
                points = d["n_elements"] * PYMUST_DATASET_PARAMS["n_voxels"] * PYMUST_DATASET_PARAMS["n_frames"]
            elif test_type == "frames":
                points = PYMUST_DATASET_PARAMS["n_elements"] * PYMUST_DATASET_PARAMS["n_voxels"] * d["n_frames"]
            total_points.append(points)

        # Add to overall collection for x-axis limits calculation
        all_total_points.extend(total_points)

        # Calculate points per second
        y_points_per_sec = []
        y_points_errors = []

        for d in data:
            pps = calculate_points_per_second(d["mean_time"], d["scaling_factor"], test_type)
            y_points_per_sec.append(pps)

            # Error propagation for points per second
            # If pps = total_points / time, then error in pps ≈ total_points * error_time / time²
            total_points_calc = pps * d["mean_time"]
            pps_error = total_points_calc * d["stddev_time"] / (d["mean_time"] ** 2)
            y_points_errors.append(pps_error)

        # Top row: Runtime scaling vs Total Points
        axes[0, col].errorbar(
            total_points,
            y_times,
            yerr=y_errors,
            marker="o",
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
        )
        axes[0, col].set_ylabel("Runtime (seconds)")
        axes[0, col].set_title(f"{title} vs Runtime")
        axes[0, col].set_xscale("log")
        axes[0, col].set_yscale("log")
        axes[0, col].grid(True, alpha=0.3)

        # Add secondary x-axis for specific parameter (top row)
        ax2_top = axes[0, col].twiny()
        ax2_top.set_xscale("log")
        secondary_axes.append(ax2_top)

        # Set tick locations and labels for specific parameter
        if len(x_values) >= 2:
            # Map total points to x_values for ticks
            ax2_top.set_xticks(total_points)
            # Format specific parameter values
            if test_type == "voxels":
                # Calculate square root dimensions for voxels
                param_labels = []
                for v in x_values:
                    # Calculate approximate square root dimensions
                    sqrt_dim = int(np.sqrt(v))
                    param_labels.append(f"{sqrt_dim}x{sqrt_dim}")
                ax2_top.set_xlabel("Grid Dimensions (Voxels)", fontsize=10)
            elif test_type == "elements":
                param_labels = [f"{int(v)}" for v in x_values]
                ax2_top.set_xlabel("Number of Receive-Elements", fontsize=10)
            elif test_type == "frames":
                param_labels = [f"{int(v)}" for v in x_values]
                ax2_top.set_xlabel("Number of Frames", fontsize=10)

            ax2_top.set_xticklabels(param_labels, rotation=45, ha="left")
        ax2_top.tick_params(axis="x", labelsize=8)

        # Bottom row: Points per second scaling vs Total Points
        axes[1, col].errorbar(
            total_points,
            y_points_per_sec,
            yerr=y_points_errors,
            marker="s",
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            color="orange",
        )
        axes[1, col].set_ylabel("Points per Second")
        axes[1, col].set_title(f"{title} vs Throughput")
        axes[1, col].set_xscale("log")
        axes[1, col].set_yscale("log")
        axes[1, col].grid(True, alpha=0.3)

        # Extend y-axis range for bottom row to show more detail
        current_ylim = axes[1, col].get_ylim()
        # Extend lower bound by factor of 10, keep upper bound with some margin
        new_ylim = (current_ylim[0] / 10, current_ylim[1] * 2)
        axes[1, col].set_ylim(new_ylim)

        # Add secondary x-axis for specific parameter (bottom row)
        ax2_bottom = axes[1, col].twiny()
        ax2_bottom.set_xscale("log")
        secondary_axes.append(ax2_bottom)

        # Set tick locations and labels for specific parameter
        if len(x_values) >= 2:
            # Map total points to x_values for ticks
            ax2_bottom.set_xticks(total_points)
            # Use same parameter labels as top row
            ax2_bottom.set_xticklabels(param_labels, rotation=45, ha="left")

            if test_type == "voxels":
                ax2_bottom.set_xlabel("Grid Dimensions (Voxels)", fontsize=10)
            elif test_type == "elements":
                ax2_bottom.set_xlabel("Number of Receive-Elements", fontsize=10)
            elif test_type == "frames":
                ax2_bottom.set_xlabel("Number of Frames", fontsize=10)

        ax2_bottom.tick_params(axis="x", labelsize=8)

        # Add annotations for scaling behavior
        if len(total_points) >= 2:
            # Calculate approximate scaling exponent using total points
            log_x = np.log10(total_points)
            log_y_time = np.log10(y_times)
            log_y_pps = np.log10(y_points_per_sec)

            # Linear fit to get scaling exponent
            time_slope = np.polyfit(log_x, log_y_time, 1)[0]
            pps_slope = np.polyfit(log_x, log_y_pps, 1)[0]

            # Add text annotation
            axes[0, col].text(
                0.05,
                0.95,
                f"Slope: ~{time_slope:.2f}",
                transform=axes[0, col].transAxes,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                verticalalignment="top",
            )

            axes[1, col].text(
                0.05,
                0.05,
                f"Slope: ~{pps_slope:.2f}",
                transform=axes[1, col].transAxes,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                verticalalignment="bottom",
            )

    # Calculate x-axis limits based on actual data
    if all_total_points:
        x_min = min(all_total_points)
        x_max = max(all_total_points)

        # Add some padding (10% on each side in log space)
        log_x_min = np.log10(x_min)
        log_x_max = np.log10(x_max)
        log_range = log_x_max - log_x_min
        log_padding = log_range * 0.1

        x_min_padded = 10 ** (log_x_min - log_padding)
        x_max_padded = 10 ** (log_x_max + log_padding)

        # Set x-axis limits for all plots
        for col in range(3):
            axes[0, col].set_xlim(x_min_padded, x_max_padded)
            axes[1, col].set_xlim(x_min_padded, x_max_padded)

        # Update secondary x-axis limits
        for ax in secondary_axes:
            ax.set_xlim(x_min_padded, x_max_padded)

    # Set shared x-axis label for bottom row
    fig.text(0.5, 0.02, "Total Points (voxels × receive-elements × frames)", ha="center", fontsize=12)

    # Get system info for overall title
    system_info = get_system_info(benchmark_data)
    gpu_info = get_gpu_info()

    # Overall title
    fig.suptitle(
        f"mach Performance Scaling\nPyMUST Rotating Disk Dataset | {system_info['cpu']} | {gpu_info}",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust layout to accommodate secondary x-axes and shared x-axis label
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.08)  # More space for secondary axes and shared label

    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Scaling plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def get_system_info(data: dict) -> dict:
    """Extract system information from benchmark data."""
    machine_info = data.get("machine_info", {})

    system_info = {
        "cpu": machine_info.get("cpu", {}).get("brand_raw", "Unknown CPU"),
        "system": f"{machine_info.get('system', 'Unknown')} {machine_info.get('release', '')}".strip(),
        "python": f"Python {machine_info.get('python_version', 'Unknown')}",
        "node": machine_info.get("node", "Unknown"),
    }

    return system_info


def get_gpu_info() -> str:
    """Try to detect GPU model using cupy."""
    try:
        import cupy as cp

        # Get GPU properties
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        return props["name"].decode("utf-8")
    except ImportError:
        pass

    return "Unknown GPU"


def print_scaling_summary(scaling_data: dict[str, list[dict]]) -> None:
    """Print summary of scaling results."""
    print("\nPerformance Scaling Summary:")
    print("=" * 60)

    for test_type, data in scaling_data.items():
        if not data:
            continue

        print(f"\n{test_type.upper()} SCALING:")
        print("-" * 30)

        for d in data:
            if test_type == "voxels":
                print(f"  {d['resolution']:.0e} resolution: {d['n_voxels']:,} voxels → {d['mean_time'] * 1000:.1f} ms")
            elif test_type == "elements":
                print(
                    f"  {d['multiplier']}x receive-elements: {d['n_elements']:,} receive-elements → {d['mean_time'] * 1000:.1f} ms"
                )
            elif test_type == "frames":
                print(f"  {d['multiplier']}x frames: {d['n_frames']:.0f} frames → {d['mean_time'] * 1000:.1f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance scaling results from pytest-benchmark JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use latest benchmark file
  %(prog)s benchmark.json                     # Use specific file
  %(prog)s --output scaling_results.png       # Save to file
        """,
    )

    parser.add_argument(
        "json_file", nargs="?", help="Path to pytest-benchmark JSON file (default: latest in .benchmarks/)"
    )

    parser.add_argument("--output", "-o", help="Output file path for the plot")

    args = parser.parse_args()

    # Determine input file
    if args.json_file:
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            return 1
    else:
        json_path = find_latest_benchmark_file()
        print(f"Using latest benchmark file: {json_path}")

    # Load and process data
    benchmark_data = load_benchmark_data(json_path)
    scaling_data = calculate_scaling_factors(benchmark_data)

    if not any(scaling_data.values()):
        print("Error: No scaling benchmark data found in JSON file")
        return 1

    # Print summary
    print_scaling_summary(scaling_data)

    # Create plots
    create_scaling_plots(scaling_data, benchmark_data, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
