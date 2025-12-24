import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Interactive Power Doppler Imaging

    This example demonstrates GPU-accelerated power Doppler beamforming using a rotating disk
    phantom dataset from [PyMUST](https://www.biomecardio.com/MUST/index.html).

    ## Dataset

    - **128-element linear array** at 5 MHz center frequency
    - **32 temporal frames** at 10 kHz PRF
    - **Rotating disk phantom** creating controlled Doppler shifts

    ## Interactive Parameters

    Adjust the sliders on the left to explore how different parameters affect power Doppler imaging:

    - **F-number**: Controls aperture size and focal characteristics
    - **Speed of Sound**: Adjust for different tissue types
    - **Tukey Alpha**: Apodization window control
    - **Channel Selection**: Sub-aperture experiments
    - **Frame Selection**: Temporal window for Doppler analysis
    - **Dynamic Range**: Optimize display contrast

    ---
    """)
    return


@app.cell
def _():
    # Import Required Libraries

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from einops import rearrange

    # Import mach modules
    from mach import wavefront
    from mach._vis import db_zero
    from mach.io.must import (
        download_pymust_doppler_data,
        extract_pymust_params,
        linear_probe_positions,
        scan_grid,
    )
    from mach.kernel import beamform

    # Check for PyMUST dependency
    try:
        import pymust
    except ImportError as err:
        raise ImportError("⚠️  PyMUST is currently required for RF-to-IQ demodulation.") from err

    # Convenience constant
    MM_PER_METER = 1000
    return (
        MM_PER_METER,
        beamform,
        db_zero,
        download_pymust_doppler_data,
        extract_pymust_params,
        linear_probe_positions,
        mo,
        np,
        plt,
        pymust,
        rearrange,
        scan_grid,
        wavefront,
    )


@app.cell
def _(download_pymust_doppler_data, extract_pymust_params):
    # Load PyMUST data (cached after first download)
    mat_data = download_pymust_doppler_data()
    params = extract_pymust_params(mat_data)
    return mat_data, params


@app.cell
def _(mat_data, params, pymust):
    # Convert RF to IQ format
    rf_data = mat_data["RF"].astype(float)
    iq_data = pymust.rf2iq(rf_data, params)
    return (iq_data,)


@app.cell
def _(linear_probe_positions, np, params, scan_grid):
    # Set up imaging geometry
    element_positions = linear_probe_positions(params["Nelements"], params["pitch"])

    # Create 2D imaging grid (±12.5 mm lateral, 10-35 mm depth)
    x = np.linspace(-12.5e-3, 12.5e-3, num=251, endpoint=True)
    y = np.array([0.0])
    z = np.linspace(10e-3, 35e-3, num=251, endpoint=True)
    grid_points = scan_grid(x, y, z)
    return element_positions, grid_points, x, z


@app.cell
def _(grid_points, np, params, wavefront):
    # Compute transmit delays for 0° plane wave
    wavefront_arrivals_s = (
        wavefront.plane(
            origin_m=np.array([0, 0, 0]),
            points_m=grid_points,
            direction=np.array([0, 0, 1]),
        )
        / params["c"]
    )
    return (wavefront_arrivals_s,)


@app.cell
def _(iq_data, np, rearrange):
    # Prepare data and extract dimensions
    iq_data_reordered = np.ascontiguousarray(
        rearrange(iq_data, "samples elements frames -> elements samples frames"), dtype=np.complex64
    )

    # Extract dimensions for slider ranges
    n_total_channels = iq_data_reordered.shape[0]
    n_total_frames = iq_data_reordered.shape[2]
    return iq_data_reordered, n_total_channels, n_total_frames


@app.cell
def _(mo, n_total_channels, n_total_frames):
    # Interactive Controls
    # --------------------
    # Create sliders for interactive parameter adjustment

    f_number = mo.ui.slider(start=0.5, stop=4.0, value=1.7, step=0.1, label="F-number", show_value=True)

    sound_speed = mo.ui.slider(
        start=1400, stop=1600, value=1540, step=10, label="Speed of Sound (m/s)", show_value=True
    )

    tukey_alpha = mo.ui.slider(
        start=0.0, stop=1.0, value=0.0, step=0.1, label="Tukey Alpha", show_value=True
    )

    channel_range = mo.ui.range_slider(
        start=0, stop=n_total_channels, value=[0, n_total_channels], step=1, label="Channel Range", show_value=True
    )

    frame_range = mo.ui.range_slider(
        start=0, stop=n_total_frames, value=[0, n_total_frames], step=1, label="Frame Range", show_value=True
    )

    vmin_db = mo.ui.slider(start=-80, stop=-10, value=-40, step=5, label="Dynamic Range (dB)", show_value=True)
    return (
        channel_range,
        f_number,
        frame_range,
        sound_speed,
        tukey_alpha,
        vmin_db,
    )


@app.cell
def _(
    MM_PER_METER,
    beamform,
    channel_range,
    db_zero,
    element_positions,
    f_number,
    frame_range,
    grid_points,
    iq_data_reordered,
    mo,
    np,
    params,
    plt,
    sound_speed,
    tukey_alpha,
    vmin_db,
    wavefront_arrivals_s,
    x,
    z,
):
    # Interactive Visualization
    # -------------------------
    # This cell reactively updates when any slider changes

    # Extract slider values
    ch_start, ch_end = channel_range.value
    fr_start, fr_end = frame_range.value

    # Slice data by channel and frame selection
    sliced_iq_data = iq_data_reordered[ch_start:ch_end, :, fr_start:fr_end]
    sliced_rx_coords = element_positions[ch_start:ch_end, :]

    # Beamform with current parameters
    result = beamform(
        channel_data=sliced_iq_data,
        rx_coords_m=sliced_rx_coords,
        scan_coords_m=grid_points,
        tx_wave_arrivals_s=wavefront_arrivals_s,
        f_number=f_number.value,
        rx_start_s=float(params["t0"]),
        sampling_freq_hz=float(params["fs"]),
        sound_speed_m_s=sound_speed.value,
        modulation_freq_hz=float(params["fc"]),
        tukey_alpha=tukey_alpha.value,
    )

    # Compute power Doppler from selected frames
    power_doppler = np.square(np.abs(result)).sum(axis=-1)
    power_doppler_2d = power_doppler.reshape(len(x), len(z))
    power_doppler_db = db_zero(power_doppler_2d)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    extent = [
        x.min() * MM_PER_METER,
        x.max() * MM_PER_METER,
        z.max() * MM_PER_METER,
        z.min() * MM_PER_METER,
    ]

    im = ax.imshow(
        power_doppler_db.T,
        cmap="hot",
        vmax=0,
        vmin=vmin_db.value,
        extent=extent,
        aspect="equal",
        origin="upper",
    )

    n_chs = ch_end - ch_start
    n_frs = fr_end - fr_start

    ax.set_title(
        f"Power Doppler: {n_chs} Channels, {n_frs} Frames\n"
        f"F-number: {f_number.value:.1f}, SoS: {sound_speed.value} m/s, Tukey α: {tukey_alpha.value:.1f}",
        fontsize=12,
    )
    ax.set_xlabel("Lateral [mm]", fontsize=11)
    ax.set_ylabel("Depth [mm]", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Magnitude [dB]", fontsize=11)

    plt.close(fig)

    # Create control panel
    controls = mo.vstack([
        mo.md("### Parameters"),
        sound_speed,
        tukey_alpha,
        mo.md("### Receive Aperture"),
        channel_range,
        f_number,
        mo.md("### Temporal Selection"),
        frame_range,
        mo.md("### Display"),
        vmin_db,
    ])

    # Display side-by-side layout
    layout = mo.hstack([controls, fig], widths=[1, 3])
    layout
    return


if __name__ == "__main__":
    app.run()
