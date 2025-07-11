#!/usr/bin/env python3
"""
Plot benchmark results from pytest-benchmark JSON files.

This script creates horizontal boxplots similar to the seaborn example at:
https://seaborn.pydata.org/examples/horizontal_boxplot.html

Usage:
    python tests/plot_benchmark.py [json_file_path] [--points-per-second] [--output output.png]

If no JSON file path is provided, uses the most recent file from .benchmarks/**/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Constants based on PyMUST rotating disk dataset
# Based on the imaging grid and data setup in tests/compare/test_pymust.py
PYMUST_DATASET_PARAMS = {
    "n_elements": 128,  # From PyMUST L7-4 linear array
    "n_frames": 32,  # Typical frame count for rotating disk dataset
    "grid_x_size": 251,  # From dx_m = 100e-6, x_range = np.arange(-1.25e-2, 1.25e-2 + dx_m, dx_m)
    "grid_z_size": 251,  # From dz_m = 100e-6, z_range = np.arange(1e-2, 3.5e-2 + dz_m, dz_m)
    "n_voxels": 251 * 251,  # 2D grid: x_grid * z_grid
    "sound_speed_m_s": 1480,
    "grid_z_max_m": 3.5e-2,
}

# Dataset or physical limits to compare against
BASELINE_RUNTIME_S = {
    # "real-time (10kHz PRI)": PYMUST_DATASET_PARAMS["n_frames"] / 10e3,
    f"speed-of-sound ({PYMUST_DATASET_PARAMS['grid_z_max_m'] * 1e3:.0f} mm)": (
        (2 * PYMUST_DATASET_PARAMS["grid_z_max_m"]) / PYMUST_DATASET_PARAMS["sound_speed_m_s"]
    )
    * PYMUST_DATASET_PARAMS["n_frames"],
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


def extract_benchmark_stats(data: dict) -> pd.DataFrame:
    """Extract benchmark statistics into a pandas DataFrame."""
    records = []

    for benchmark in data["benchmarks"]:
        # Clean up the name for display
        name = benchmark["name"].removeprefix("test_")
        # Remove substring _benchmark
        name = name.replace("_benchmark", "")

        # Extract timing data
        stats = benchmark["stats"]

        record = {
            "name": name,
            "group": benchmark.get("group", "default"),
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "median": stats["median"],
            "q1": stats["q1"],
            "q3": stats["q3"],
            "data": stats["data"],  # Raw timing data for boxplot
            "fullname": benchmark["fullname"],
        }
        records.append(record)

    return pd.DataFrame(records)


def calculate_points_per_second(timing_seconds: float) -> float:
    """Calculate effective points per second for the given timing."""
    total_points = (
        PYMUST_DATASET_PARAMS["n_elements"] * PYMUST_DATASET_PARAMS["n_voxels"] * PYMUST_DATASET_PARAMS["n_frames"]
    )
    return total_points / timing_seconds


def add_baseline_references(df: pd.DataFrame, use_points_per_second: bool = False) -> pd.DataFrame:
    """Add baseline reference entries to the DataFrame."""
    baseline_records = []

    for name, timing_seconds in BASELINE_RUNTIME_S.items():
        # Store the raw timing data in the same format as benchmark data
        record = {
            "name": name,
            "group": "baseline",
            "min": timing_seconds,
            "max": timing_seconds,
            "mean": timing_seconds,
            "median": timing_seconds,
            "q1": timing_seconds,
            "q3": timing_seconds,
            "data": [timing_seconds],  # Raw timing data in seconds
            "fullname": f"baseline::{name}",
        }
        baseline_records.append(record)

    baseline_df = pd.DataFrame(baseline_records)
    return pd.concat([df, baseline_df], ignore_index=True)


def create_boxplot_data(df: pd.DataFrame, use_points_per_second: bool = False) -> tuple[pd.DataFrame, str]:
    """Create data suitable for seaborn boxplot."""
    plot_data = []

    for _, row in df.iterrows():
        timing_data = row["data"]  # This is always in seconds

        if use_points_per_second:
            # Convert each timing value to points per second
            values = [calculate_points_per_second(t) for t in timing_data]
            unit = "points/second"
        else:
            # Keep as seconds
            values = list(timing_data)
            unit = "time (s)"

        # Add each measurement as a separate row
        for value in values:
            plot_data.append({"method": row["name"], "group": row["group"], unit: value})

    return pd.DataFrame(plot_data), unit


def plot_benchmark_results(
    df: pd.DataFrame,
    data: dict,
    use_points_per_second: bool = False,
    use_bar_chart: bool = True,
    output_path: Optional[str] = None,
    show_values: bool = True,
    short_title: bool = False,
    linear_scale: bool = False,
    broken_axis: bool = False,
) -> None:
    """Create horizontal plot of benchmark results (bar chart or boxplot)."""

    # Add baseline references
    df_with_baselines = add_baseline_references(df, use_points_per_second)

    # Set up the plot style
    sns.set_style("ticks")
    sns.set_palette("Set2")

    if broken_axis and use_bar_chart and linear_scale:
        # Special handling for broken axis - create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
        fig.subplots_adjust(wspace=0.05)  # adjust space between Axes

        # Calculate statistics for bar chart
        method_stats = []
        for _, row in df_with_baselines.iterrows():
            timing_data = row["data"]

            if use_points_per_second:
                values = [calculate_points_per_second(t) for t in timing_data]
            else:
                values = list(timing_data)

            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)

            method_stats.append({
                "method": row["name"],
                "group": row["group"],
                "mean": mean_val,
                "std": std_val,
                "median": np.median(values),
            })

        # Convert to DataFrame and sort
        stats_df = pd.DataFrame(method_stats)
        stats_df = stats_df.sort_values("mean")

        # Find the breakpoint - typically the largest outlier
        max_val = stats_df["mean"].max()
        second_max = stats_df["mean"].nlargest(2).iloc[1]

        # Set break point between the largest and second largest
        break_point = second_max + (max_val - second_max) * 0.1

        # Create colors
        colors = sns.color_palette("Set2", len(stats_df))

        # Plot ALL data on BOTH axes
        bars1 = ax1.barh(
            range(len(stats_df)),
            stats_df["mean"],
            xerr=stats_df["std"],
            height=0.6,
            capsize=3,
            color=colors,
            alpha=0.8,
            error_kw={"alpha": 0.6},
        )

        bars2 = ax2.barh(
            range(len(stats_df)),
            stats_df["mean"],
            xerr=stats_df["std"],
            height=0.6,
            capsize=3,
            color=colors,
            alpha=0.8,
            error_kw={"alpha": 0.6},
        )

        # Set y-axis labels on both axes
        ax1.set_yticks(range(len(stats_df)))
        ax1.set_yticklabels(stats_df["method"])
        ax2.set_yticks(range(len(stats_df)))
        ax2.set_yticklabels(stats_df["method"])

        # Set different x-axis limits for each panel
        # Left panel: focus on smaller values (regular data)
        ax1.set_xlim(0, break_point)
        # Right panel: focus on larger values (outliers)
        ax2.set_xlim(break_point * 0.9, max_val * 1.3)

        # Add value annotations if requested
        if show_values:
            for i, (_, row) in enumerate(stats_df.iterrows()):
                value = row["median"]

                # Format the value appropriately
                if use_points_per_second:
                    value_text = f"{value:.2e}"
                else:
                    if value >= 1:
                        value_text = f"{value:.3f} s"
                    else:
                        value_text = f"{value * 1000:.1f} ms"

                # Determine which panel to put the annotation on
                if row["mean"] <= break_point:
                    # Small values go on left panel
                    x_pos = row["mean"] + row["std"] * 1.2
                    if x_pos < break_point * 0.95:  # Make sure annotation fits within panel
                        ax1.text(
                            x_pos,
                            i,
                            value_text,
                            verticalalignment="center",
                            fontsize=9,
                            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                        )
                else:
                    # Large values go on right panel
                    x_pos = row["mean"] + row["std"] * 1.2
                    ax2.text(
                        x_pos,
                        i,
                        value_text,
                        verticalalignment="center",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    )

        # Hide spines between axes
        ax1.spines.right.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()

        # Add proper broken axis slashes following matplotlib example style
        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False
        )
        # For horizontal broken axis, we need to place slashes on the right edge of left panel
        # and left edge of right panel
        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

        # Set unit labels
        unit_label = "points/second" if use_points_per_second else "time (s)"
        fig.text(0.5, 0.02, unit_label, ha="center", fontsize=12)

        # Add grids
        ax1.grid(True, alpha=0.3, axis="x")
        ax2.grid(True, alpha=0.3, axis="x")

        # Adjust x-axis limits to prevent overlapping labels
        if show_values:
            # Left panel: ensure space for annotations
            ax1.set_xlim(0, break_point * 0.95)
            # Right panel: ensure space for annotations
            ax2.set_xlim(break_point * 0.9, max_val * 1.4)

        # Improve x-axis tick spacing to prevent overlap
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))

        # Get system info for title
        system_info = get_system_info(data)
        gpu_info = get_gpu_info()

        # Set title on the figure
        if short_title:
            title_text = "Beamforming the PyMUST Rotating Disk Dataset"
        else:
            title_text = (
                f"Beamforming Performance Comparison\n"
                f"PyMUST Rotating Disk Dataset "
                f"({PYMUST_DATASET_PARAMS['n_elements']} elements, "
                f"{PYMUST_DATASET_PARAMS['n_voxels']:,} voxels, "
                f"{PYMUST_DATASET_PARAMS['n_frames']} frames)\n"
                f"{system_info['cpu']} | {gpu_info} | {system_info['system']}"
            )
        fig.suptitle(title_text, fontsize=12)

    else:
        # Original single-axis plotting
        # Create figure - make it wider if showing values
        fig_width = 8 if show_values else 6
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        if use_bar_chart:
            # Calculate statistics for bar chart
            method_stats = []
            for _, row in df_with_baselines.iterrows():
                timing_data = row["data"]

                if use_points_per_second:
                    values = [calculate_points_per_second(t) for t in timing_data]
                else:
                    values = list(timing_data)

                # Calculate statistics
                mean_val = np.mean(values)
                std_val = np.std(values)

                method_stats.append({
                    "method": row["name"],
                    "group": row["group"],
                    "mean": mean_val,
                    "std": std_val,
                    "median": np.median(values),
                })

            # Convert to DataFrame and sort
            stats_df = pd.DataFrame(method_stats)
            stats_df = stats_df.sort_values("mean")

            # Create horizontal bar chart
            colors = sns.color_palette("Set2", len(stats_df))
            bars = ax.barh(
                range(len(stats_df)),
                stats_df["mean"],
                xerr=stats_df["std"],
                height=0.6,
                capsize=3,
                color=colors,
                alpha=0.8,
                error_kw={"alpha": 0.6},
            )

            # Set y-axis labels
            ax.set_yticks(range(len(stats_df)))
            ax.set_yticklabels(stats_df["method"])

            # Set unit label for x-axis
            unit_label = "points/second" if use_points_per_second else "time (s)"
            ax.set_xlabel(unit_label)

            # Add value annotations if requested
            if show_values:
                for i, (_, row) in enumerate(stats_df.iterrows()):
                    value = row["median"]  # Use median for consistency with boxplot

                    # Format the value appropriately
                    if use_points_per_second:
                        value_text = f"{value:.2e}"
                    else:
                        if value >= 1:
                            value_text = f"{value:.3f} s"
                        else:
                            value_text = f"{value * 1000:.1f} ms"

                    # Position text closer to the bar end (mean + std for error bar end)
                    x_pos = row["mean"] + row["std"] * 1.1  # Small offset from error bar

                    ax.text(
                        x_pos,
                        i,
                        value_text,
                        verticalalignment="center",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    )

        else:
            # Original boxplot code
            plot_df, unit_label = create_boxplot_data(df_with_baselines, use_points_per_second)

            # Calculate mean values and sort methods
            method_stats = plot_df.groupby("method")[unit_label].agg(["mean", "median"]).sort_values("mean")
            method_order = method_stats.index

            # Create boxplot with ordered methods
            sns.boxplot(
                data=plot_df,
                y="method",
                x=unit_label,
                ax=ax,
                orient="h",
                order=method_order,
                whis=100,  # extend whiskers instead of showing fliers
            )

            # Add value annotations if requested
            if show_values:
                # Get the maximum x-value for positioning text
                x_max = plot_df[unit_label].max()

                # Add text annotations showing median values
                for i, method in enumerate(method_order):
                    median_val = method_stats.loc[method, "median"]

                    # Format the value appropriately
                    if use_points_per_second:
                        value_text = f"{median_val:.2e}"
                    else:
                        if median_val >= 1:
                            value_text = f"{median_val:.3f} s"
                        else:
                            value_text = f"{median_val * 1000:.1f} ms"

                    # Position text to the right of the plot
                    ax.text(
                        x_max * 1.5,
                        i,
                        value_text,
                        verticalalignment="center",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    )

        # Get system info for title
        system_info = get_system_info(data)
        gpu_info = get_gpu_info()

        # Customize the plot
        if short_title:
            title_lines = [
                "Beamforming the PyMUST Rotating Disk Dataset",
            ]
        else:
            title_lines = [
                "Beamforming Performance Comparison",
                f"PyMUST Rotating Disk Dataset "
                f"({PYMUST_DATASET_PARAMS['n_elements']} elements, "
                f"{PYMUST_DATASET_PARAMS['n_voxels']:,} voxels, "
                f"{PYMUST_DATASET_PARAMS['n_frames']} frames)",
                f"{system_info['cpu']} | {gpu_info} | {system_info['system']}",
            ]
        ax.set_title("\n".join(title_lines))

        # Set scale (linear or log)
        if not linear_scale:
            ax.set_xscale("log")

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis="x")

        # Extend x-axis limits if showing values to make room for text
        if show_values and use_bar_chart:
            # For bar chart, extend based on the rightmost annotation
            xlim = ax.get_xlim()
            max_x_pos = max((row["mean"] + row["std"]) for _, row in stats_df.iterrows())
            ax.set_xlim(xlim[0], max_x_pos * 2.5)  # Give some room for annotations
        elif show_values:
            # Original boxplot behavior
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0], xlim[1] * 3)  # Extend right limit

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def filter_benchmarks_by_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Filter benchmarks by group, excluding specific tests."""
    # Filter by group
    filtered_df = df[df["group"] == group].copy()

    # Exclude mach_from_cpu
    filtered_df = filtered_df[~filtered_df["name"].str.contains("mach_from_cpu", na=False)]

    return filtered_df


def print_summary_stats(df: pd.DataFrame, use_points_per_second: bool = False) -> None:
    """Print summary statistics."""
    print("\nBenchmark Summary:")
    print("=" * 50)

    for _, row in df.iterrows():
        name = row["name"]
        if use_points_per_second:
            mean_val = calculate_points_per_second(row["mean"])
            unit = "points/sec"
            print(f"{name:30s}: {mean_val:15.2e} {unit}")
        else:
            mean_val = row["mean"] * 1000  # Convert to ms
            unit = "ms"
            print(f"{name:30s}: {mean_val:10.3f} {unit}")


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
    """Try to detect GPU model using jax or cupy."""

    try:
        import cupy as cp

        # Get GPU properties
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        return props["name"].decode("utf-8")
    except ImportError:
        pass

    return "Unknown GPU"


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from pytest-benchmark JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use latest benchmark file with bar chart
  %(prog)s benchmark.json                     # Use specific file
  %(prog)s --points-per-second               # Show points/sec instead of runtime
  %(prog)s --output results.png              # Save to file
  %(prog)s --group all                       # Include all benchmark groups
  %(prog)s --no-values                       # Don't show value annotations
  %(prog)s --box-plot                        # Use box plot instead of bar chart
  %(prog)s --short-title                     # Use abbreviated title
  %(prog)s --linear-scale                    # Use linear scale instead of log scale
  %(prog)s --broken-axis                     # Use broken axis for outliers (requires --linear-scale)
        """,
    )

    parser.add_argument(
        "json_file", nargs="?", help="Path to pytest-benchmark JSON file (default: latest in .benchmarks/)"
    )

    parser.add_argument("--points-per-second", action="store_true", help="Plot points per second instead of runtime")

    parser.add_argument("--output", "-o", help="Output file path for the plot")

    parser.add_argument(
        "--group",
        "-g",
        default="doppler_disk",
        help='Benchmark group to include (default: doppler_disk, use "all" for all groups)',
    )

    parser.add_argument("--no-values", action="store_true", help="Don't show value annotations on the plot")

    parser.add_argument(
        "--box-plot", action="store_true", help="Use box plot instead of bar chart (default: bar chart)"
    )

    parser.add_argument("--short-title", action="store_true", help="Use abbreviated title")

    parser.add_argument(
        "--linear-scale", action="store_true", help="Use linear scale instead of log scale (default: log scale)"
    )

    parser.add_argument(
        "--broken-axis",
        action="store_true",
        help="Use broken axis for outliers (requires --linear-scale and bar chart)",
    )

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
    data = load_benchmark_data(json_path)
    df = extract_benchmark_stats(data)

    if df.empty:
        raise RuntimeError("Error: No benchmark data found in JSON file")

    # Filter by group if specified
    if args.group != "all":
        df = filter_benchmarks_by_group(df, args.group)
        if df.empty:
            print(f"Warning: No benchmarks found in group '{args.group}'")
            print("Available groups:", df["group"].unique().tolist())
            return 1

    # Print summary
    print_summary_stats(df, args.points_per_second)

    # Create plot
    plot_benchmark_results(
        df,
        data,
        args.points_per_second,
        use_bar_chart=not args.box_plot,  # Default to bar chart unless --box-plot is specified
        output_path=args.output,
        show_values=not args.no_values,
        short_title=args.short_title,
        linear_scale=args.linear_scale,
        broken_axis=args.broken_axis,
    )


if __name__ == "__main__":
    exit(main())
