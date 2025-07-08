#!/usr/bin/env python3
"""
Plot performance scaling results from pytest-benchmark JSON files.

This script creates a 2x3 subplot grid showing:
- Top row: Runtime scaling (seconds)
- Bottom row: Points-per-second scaling
- Columns: Voxels, Elements, Frames

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
        # Handle fractional and integer multipliers
        if "1/32x_frames" in param_string:
            return 1 / 32
        elif "1/8x_frames" in param_string:
            return 1 / 8
        else:
            match = re.search(r"([0-9]+)x_frames", param_string)
            if match:
                return int(match.group(1))

    return None


def calculate_scaling_factors(benchmark_data: dict) -> dict[str, list[dict]]:
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

    # Sort by scaling factor for proper plotting
    for key in scaling_data:
        scaling_data[key].sort(key=lambda x: x["scaling_factor"])

    return scaling_data


def calculate_points_per_second(time_seconds: float, scaling_factor: float, test_type: str) -> float:
    """Calculate effective points per second for the given timing and scaling."""
    if test_type == "voxels":
        # Points = elements × voxels × frames
        total_points = (
            PYMUST_DATASET_PARAMS["n_elements"]
            * PYMUST_DATASET_PARAMS["n_voxels"]
            * scaling_factor
            * PYMUST_DATASET_PARAMS["n_frames"]
        )
    elif test_type == "elements":
        # Points = elements × voxels × frames
        total_points = (
            PYMUST_DATASET_PARAMS["n_elements"]
            * scaling_factor
            * PYMUST_DATASET_PARAMS["n_voxels"]
            * PYMUST_DATASET_PARAMS["n_frames"]
        )
    elif test_type == "frames":
        # Points = elements × voxels × frames
        total_points = (
            PYMUST_DATASET_PARAMS["n_elements"]
            * PYMUST_DATASET_PARAMS["n_voxels"]
            * PYMUST_DATASET_PARAMS["n_frames"]
            * scaling_factor
        )
    else:
        return 0

    return total_points / time_seconds


def create_scaling_plots(
    scaling_data: dict[str, list[dict]], benchmark_data: dict, output_path: Optional[str] = None
) -> None:
    """Create 2x3 subplot grid showing scaling performance."""

    # Set up the plot style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Test types and their display names
    test_types = [
        ("voxels", "Number of Voxels", "n_voxels"),
        ("elements", "Number of Elements", "n_elements"),
        ("frames", "Number of Frames", "n_frames"),
    ]

    for col, (test_type, title, x_param) in enumerate(test_types):
        data = scaling_data[test_type]

        if not data:
            continue

        # Extract data for plotting
        x_values = [d[x_param] for d in data]
        y_times = [d["mean_time"] for d in data]
        y_errors = [d["stddev_time"] for d in data]

        # Calculate points per second
        y_points_per_sec = []
        y_points_errors = []

        for d in data:
            pps = calculate_points_per_second(d["mean_time"], d["scaling_factor"], test_type)
            y_points_per_sec.append(pps)

            # Error propagation for points per second
            # If pps = total_points / time, then error in pps ≈ total_points * error_time / time²
            total_points = pps * d["mean_time"]
            pps_error = total_points * d["stddev_time"] / (d["mean_time"] ** 2)
            y_points_errors.append(pps_error)

        # Top row: Runtime scaling
        axes[0, col].errorbar(
            x_values, y_times, yerr=y_errors, marker="o", linewidth=2, markersize=8, capsize=5, capthick=2
        )
        axes[0, col].set_xlabel(title)
        axes[0, col].set_ylabel("Runtime (seconds)")
        axes[0, col].set_title(f"{title} vs Runtime")
        axes[0, col].set_xscale("log")
        axes[0, col].set_yscale("log")
        axes[0, col].grid(True, alpha=0.3)

        # Bottom row: Points per second scaling
        axes[1, col].errorbar(
            x_values,
            y_points_per_sec,
            yerr=y_points_errors,
            marker="s",
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            color="orange",
        )
        axes[1, col].set_xlabel(title)
        axes[1, col].set_ylabel("Points per Second")
        axes[1, col].set_title(f"{title} vs Throughput")
        axes[1, col].set_xscale("log")
        axes[1, col].set_yscale("log")
        axes[1, col].grid(True, alpha=0.3)

        # Add annotations for scaling behavior
        if len(x_values) >= 2:
            # Calculate approximate scaling exponent
            log_x = np.log10(x_values)
            log_y_time = np.log10(y_times)
            log_y_pps = np.log10(y_points_per_sec)

            # Linear fit to get scaling exponent
            time_slope = np.polyfit(log_x, log_y_time, 1)[0]
            pps_slope = np.polyfit(log_x, log_y_pps, 1)[0]

            # Add text annotation
            axes[0, col].text(
                0.05,
                0.95,
                f"Scaling: ~{time_slope:.2f}",
                transform=axes[0, col].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            axes[1, col].text(
                0.05,
                0.05,
                f"Scaling: ~{pps_slope:.2f}",
                transform=axes[1, col].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="bottom",
            )

    # Get system info for overall title
    system_info = get_system_info(benchmark_data)
    gpu_info = get_gpu_info()

    # Overall title
    fig.suptitle(
        f"MACH Beamformer Performance Scaling\nPyMUST Rotating Disk Dataset | {system_info['cpu']} | {gpu_info}",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Scaling plot saved to: {output_path}")
    else:
        plt.show()


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
                print(f"  {d['multiplier']}x elements: {d['n_elements']:,} elements → {d['mean_time'] * 1000:.1f} ms")
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
