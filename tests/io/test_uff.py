"""Test the UFF data loader."""

from pathlib import Path
from typing import Optional

import pytest
from pyuff_ustb import ChannelData, Scan

from mach._array_api import ArrayAPIConformant
from mach._vis import db_zero, plot_slice
from mach.experimental import beamform
from mach.io.uff import create_beamforming_setup, create_single_transmit_beamforming_setup


@pytest.mark.filterwarnings("ignore:array is not contiguous, rearranging will add latency:UserWarning")
def test_picmus_phantom_resolution(
    picmus_phantom_resolution_channel_data: ChannelData,
    picmus_phantom_resolution_scan: Scan,
    output_dir: Optional[Path],
):
    """Test the Picmus phantom resolution UFF data."""
    assert picmus_phantom_resolution_channel_data is not None
    assert picmus_phantom_resolution_scan is not None

    beamform_kwargs = create_beamforming_setup(
        picmus_phantom_resolution_channel_data,
        picmus_phantom_resolution_scan,
    )

    result = beamform(**beamform_kwargs)

    assert isinstance(result, ArrayAPIConformant)

    if output_dir is not None:
        single_slice = result.reshape(
            picmus_phantom_resolution_scan.x_axis.size, picmus_phantom_resolution_scan.z_axis.size
        )
        fig = plot_slice(
            bm_slice=db_zero(single_slice.T),
            lats=picmus_phantom_resolution_scan.x_axis.flat,
            deps=picmus_phantom_resolution_scan.z_axis.flat,
            angle=0,  # Multi-transmit compounded result
        )
        fig.savefig(output_dir / "picmus_phantom_resolution.png")


@pytest.mark.filterwarnings("ignore:array is not contiguous, rearranging will add latency:UserWarning")
def test_picmus_phantom_resolution_single_transmit(
    picmus_phantom_resolution_channel_data: ChannelData,
    picmus_phantom_resolution_scan: Scan,
    output_dir: Optional[Path],
):
    """Test the Picmus phantom resolution UFF data with single transmit (backwards compatibility)."""
    assert picmus_phantom_resolution_channel_data is not None
    assert picmus_phantom_resolution_scan is not None

    wave_index = 37
    beamform_kwargs = create_single_transmit_beamforming_setup(
        picmus_phantom_resolution_channel_data,
        picmus_phantom_resolution_scan,
        wave_index=wave_index,
    )

    # Use the single-transmit kernel for backward compatibility
    from mach.kernel import beamform as single_beamform

    result = single_beamform(**beamform_kwargs)

    assert isinstance(result, ArrayAPIConformant)

    if output_dir is not None:
        single_slice = result[..., 0].reshape(  # Single frame result
            picmus_phantom_resolution_scan.x_axis.size, picmus_phantom_resolution_scan.z_axis.size
        )
        fig = plot_slice(
            bm_slice=db_zero(single_slice.T),
            lats=picmus_phantom_resolution_scan.x_axis.flat,
            deps=picmus_phantom_resolution_scan.z_axis.flat,
            angle=wave_index,
        )
        fig.savefig(output_dir / "picmus_phantom_resolution_single_transmit.png")
