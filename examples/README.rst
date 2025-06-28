Examples Gallery
================

This directory contains tutorial examples demonstrating mach capabilities.
Each example is designed to work with sphinx-gallery for automatic documentation generation.

Available Examples
------------------

- **doppler.py**: Power Doppler beamforming tutorial using PyMUST rotating disk data.
  Demonstrates GPU-accelerated beamforming for motion detection applications.

- **plane_wave_compound.py**: Plane wave compounding with PICMUS challenge data.
  Demonstrates coherent compounding of multiple plane wave angles for improved image quality.

Dependencies
------------

**Additional common dependencies:**

- ``matplotlib``: Required for visualizations
- ``numpy``: Basic array operations

**Example-specific dependencies:**

- **doppler.py** requires ``pymust`` to load example data and demodulate RF to IQ (``pip install pymust``)
- **plane_wave_compound.py** requires ``pyuff-ustb`` to load example data (``pip install pyuff-ustb``)
