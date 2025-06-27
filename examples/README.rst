Examples Gallery
================

This directory contains tutorial examples demonstrating mach capabilities.
Each example is designed to work with sphinx-gallery for automatic documentation generation
and can also be converted to Jupyter notebooks.

Available Examples
------------------

- **doppler.py**: Power Doppler beamforming tutorial using PyMUST rotating disk data.
  Demonstrates GPU-accelerated beamforming for motion detection applications.

- **plane_wave_compound.py**: Plane wave compounding with PICMUS challenge data.
  Demonstrates coherent compounding of multiple plane wave angles for improved image quality.

Sphinx-Gallery Integration
--------------------------

These examples follow sphinx-gallery conventions:

- Docstring with reStructuredText formatting for the main description
- ``# %%`` comments to separate code cells
- Proper attribution to external datasets and references
- Educational narrative with clear learning objectives

To build the documentation with sphinx-gallery::

    make docs

Jupyter Notebook Conversion
---------------------------

To convert any example to a Jupyter notebook::

    pip install jupytext
    jupytext --to notebook examples/*.py

Dependencies
------------

**Common dependencies:**
- **matplotlib**: Required for all visualizations
- **numpy**: Basic array operations
- **einops**: Required for array reshaping operations

**Example-specific dependencies:**

For **doppler.py**:
- **pymust**: Required for PyMUST data loading and RF-to-IQ conversion

For **plane_wave_compound.py**:
- **pyuff-ustb**: Required for UFF data loading (``pip install pyuff-ustb``)

Attribution
-----------

- **PyMUST/MUST**: Garcia, D. MUST: An Open Platform for Ultrasound Research. https://www.biomecardio.com/MUST/index.html
- **PICMUS** Liebgott, H. et al. Plane-Wave Imaging Challenge in Medical Ultrasound. https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/
- **Inspired by**: ultraspy tutorials (https://ultraspy.readthedocs.io/) and vbeam examples (https://github.com/magnusdk/vbeam/)
