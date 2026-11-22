Gallery
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

**Recommended Installation:**

To run all examples without dependency issues, we recommend installing mach with all optional dependencies::

    pip install mach-beamform[all]

This includes all visualization, data loading, and example-specific dependencies.

**Additional common dependencies:**

- ``matplotlib``: Required for visualizations
- ``numpy``: Basic array operations

**Example-specific dependencies:**

- **doppler.py** requires ``pymust`` to load example data and demodulate RF to IQ (``pip install pymust``)
- **plane_wave_compound.py** requires ``pyuff-ustb`` to load example data (``pip install pyuff-ustb``)

Async batch benchmark
---------------------

Run the four-condition comparison on CUDA with CuPy and PyMUST installed::

    python benchmarks/async_batched_beamform.py --output async-benchmark.json

The benchmark compares synchronous and asynchronous launches with one and two
device slots. It starts with prepared IQ in pinned host memory and ends after
the GPU power reduction. This applies to both prerecorded batches and batches
prepared during live acquisition. Data loading and RF-to-IQ conversion are
outside the timed interval. Use ``--verify-only`` to check each batch against
the synchronous reference without running the timed trials.
