import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Interactive Plane Wave Compounding

    This example demonstrates GPU-accelerated plane wave compounding using data from the
    [PICMUS challenge](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/)
    (Plane-wave Imaging Challenge in Medical Ultrasound).

    ## Dataset

    - **128-element linear array** at 5.2 MHz center frequency
    - **75 plane wave transmits** at angles from -16° to +16°
    - **Point targets and cysts** for resolution and contrast assessment

    ## Interactive Parameters

    Adjust the sliders on the left to explore how different beamforming parameters
    affect image quality:

    - **F-number**: Controls aperture size and focal characteristics
    - **Speed of Sound**: Adjust for different tissue types
    - **Channel/Plane Wave Selection**: Sub-aperture beamforming experiments
    - **Dynamic Range**: Optimize display contrast

    ---
    """)
    return


@app.cell
def _():
    # Import Required Libraries
    # -------------------------
    # Let's start by importing the necessary libraries.

    import hashlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Import mach modules
    from mach import experimental
    from mach._vis import db_zero
    from mach.io.uff import create_beamforming_setup
    from mach.io.utils import cached_download

    try:
        from pyuff_ustb import Uff
    except ImportError as err:
        raise ImportError(
            "⚠️  pyuff_ustb is required for UFF data loading. Install with: pip install pyuff-ustb"
        ) from err

    # Convenience constants
    MM_PER_METER = 1000
    return (
        MM_PER_METER,
        Uff,
        cached_download,
        create_beamforming_setup,
        db_zero,
        experimental,
        hashlib,
        mo,
        np,
        plt,
    )


@app.cell
def _(cached_download, hashlib):
    # Download PICMUS Challenge Dataset (runs once, cached)
    url = "http://www.ustb.no/datasets/PICMUS_experiment_resolution_distortion.uff"
    uff_path = cached_download(
        url,
        expected_size=145_518_524,
        expected_hash="c93af0781daeebf771e53a42a629d4f311407410166ef2f1d227e9d2a1b8c641",
        digest=hashlib.sha256,
        filename="PICMUS_experiment_resolution_distortion.uff",
    )
    return (uff_path,)


@app.cell
def _(Uff, np, uff_path):
    # Load UFF data structure
    uff_file = Uff(str(uff_path))
    channel_data = uff_file.read("/channel_data")
    scan = uff_file.read("/scan")
    return channel_data, scan


@app.cell
def _(channel_data, create_beamforming_setup, np, scan):
    # Prepare base beamforming parameters
    base_kwargs = create_beamforming_setup(
        channel_data=channel_data,
        scan=scan,
        f_number=1.7,
    )

    # Extract dimensions for slider ranges
    n_total_transmits = base_kwargs["channel_data"].shape[0]
    n_total_channels = base_kwargs["channel_data"].shape[1]
    angles_deg_all = [np.rad2deg(wave.source.azimuth) for wave in channel_data.sequence]
    return angles_deg_all, base_kwargs, n_total_channels, n_total_transmits


@app.cell
def _(mo, n_total_channels, n_total_transmits):
    # Interactive Controls
    # --------------------
    # Create sliders for interactive parameter adjustment

    f_number = mo.ui.slider(start=0.5, stop=4.0, value=1.7, step=0.1, label="F-number", show_value=True)

    vmin_db = mo.ui.slider(start=-80, stop=-10, value=-40, step=5, label="Dynamic Range (dB)", show_value=True)

    channel_range = mo.ui.range_slider(
        start=0, stop=n_total_channels, value=[0, n_total_channels], step=1, label="Channel Range", show_value=True
    )

    pw_range = mo.ui.range_slider(
        start=0, stop=n_total_transmits, value=[0, n_total_transmits], step=1, label="Plane Wave Range", show_value=True
    )

    sound_speed = mo.ui.slider(
        start=1400, stop=1600, value=1540, step=10, label="Speed of Sound (m/s)", show_value=True
    )
    return channel_range, f_number, pw_range, sound_speed, vmin_db


@app.cell
def _(
    MM_PER_METER,
    angles_deg_all,
    base_kwargs,
    channel_range,
    db_zero,
    experimental,
    f_number,
    mo,
    plt,
    pw_range,
    scan,
    sound_speed,
    vmin_db,
):
    # Interactive Visualization
    # -------------------------
    # This cell reactively updates when any slider changes

    # Extract slider values
    ch_start, ch_end = channel_range.value
    pw_start, pw_end = pw_range.value

    # Slice the data based on user selections
    sliced_channel_data = base_kwargs["channel_data"][pw_start:pw_end, ch_start:ch_end, :, :]
    sliced_rx_coords = base_kwargs["rx_coords_m"][ch_start:ch_end, :]
    sliced_tx_arrivals = base_kwargs["tx_wave_arrivals_s"][pw_start:pw_end, :]

    # Update beamforming kwargs with current slider values
    current_kwargs = {
        "channel_data": sliced_channel_data,
        "rx_coords_m": sliced_rx_coords,
        "scan_coords_m": base_kwargs["scan_coords_m"],
        "tx_wave_arrivals_s": sliced_tx_arrivals,
        "out": None,
        "f_number": f_number.value,
        "sampling_freq_hz": base_kwargs["sampling_freq_hz"],
        "sound_speed_m_s": sound_speed.value,
        "modulation_freq_hz": base_kwargs["modulation_freq_hz"],
        "rx_start_s": base_kwargs["rx_start_s"],
    }

    # Perform beamforming and compounding
    result = experimental.beamform(**current_kwargs)

    # Reshape to 2D grid
    grid_shape = (scan.x_axis.size, scan.z_axis.size)
    beamformed_image = result.reshape(grid_shape)

    # Convert to dB scale
    bmode_db = db_zero(beamformed_image)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    extent = [
        scan.x_axis.min() * MM_PER_METER,
        scan.x_axis.max() * MM_PER_METER,
        scan.z_axis.max() * MM_PER_METER,
        scan.z_axis.min() * MM_PER_METER,
    ]

    im = ax.imshow(
        bmode_db.T,
        cmap="gray",
        vmin=vmin_db.value,
        vmax=0,
        extent=extent,
        aspect="equal",
        origin="upper",
        interpolation="nearest",
    )

    # Get angle range for selected plane waves
    selected_angles = angles_deg_all[pw_start:pw_end]
    n_pws = pw_end - pw_start
    n_chs = ch_end - ch_start

    ax.set_title(
        f"PICMUS: {n_pws} Plane Waves ({min(selected_angles):.0f}° to {max(selected_angles):.0f}°), "
        f"{n_chs} Channels\n"
        f"F-number: {f_number.value:.1f}, SoS: {sound_speed.value} m/s",
        fontsize=12,
    )
    ax.set_xlabel("Lateral Distance [mm]", fontsize=11)
    ax.set_ylabel("Depth [mm]", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Magnitude [dB]", fontsize=11)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.close(fig)

    # Create control panel
    controls = mo.vstack([
        mo.md("### Parameters"),
        sound_speed,
        pw_range,
        mo.md("### Receive Aperture"),
        channel_range,
        f_number,
        mo.md("### Display"),
        vmin_db,
    ])

    # Display side-by-side layout
    layout = mo.hstack([controls, fig], widths=[1, 3])
    layout
    return


if __name__ == "__main__":
    app.run()
