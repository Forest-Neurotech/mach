"""
CUDA-accelerated 3D ultrasound beamforming with GPU array support.

This package provides GPU-accelerated beamforming functionality for ultrasound data.
It supports input arrays from NumPy, CuPy, and JAX through the DLPack protocol.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mach")
except PackageNotFoundError:
    # Update via bump-my-version, not manually
    __version__ = "0.0.2"
