# API Documentation Summary

This document summarizes the changes made to add comprehensive auto-generated API documentation to the Mach project with individual pages for each function.

## Changes Made

### 1. Comprehensive API Coverage
- Generated documentation for ALL modules in the mach package
- Created individual pages for each function (like NumPy's documentation style)
- Organized functions by category and module

### 2. Added `numpydoc` dependency
- Added `numpydoc>=1.5.0` to the `docs` dependency group in `pyproject.toml`

### 3. Updated Sphinx configuration (`docs/conf.py`)
- Added `numpydoc` to the extensions list
- Added numpydoc configuration options:
  - `numpydoc_show_class_members = False`
  - `numpydoc_xref_param_type = True`
  - `numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}`
  - `numpydoc_validation_checks = {"all", "GL08", "SA01", "EX01"}`

### 4. Created comprehensive API documentation structure (`docs/api.rst`)
- Individual pages for every function using autosummary with `:toctree: generated/`
- Organized by functional categories:
  - **Core Beamforming**: `kernel.beamform`
  - **Wavefront Models**: `wavefront.plane`, `wavefront.spherical`
  - **Geometry Utilities**: coordinate conversion functions
  - **Experimental Features**: experimental beamforming and device utilities
  - **Visualization**: plotting and debugging functions  
  - **I/O Utilities**: UFF file format, PyMUST data, and general I/O functions
- Module overview with brief descriptions

### 5. Updated main documentation (`docs/index.rst`)
- Added API Reference section to the main toctree

### 6. Fixed docstring formatting
- Updated docstrings in `kernel.py` and `wavefront.py` to use proper NumPy style:
  - Changed `Parameters:` to `Parameters` (without colon)
  - Changed `Returns:` to `Returns` (without colon)
  - Changed `Notes:` to `Notes` (without colon)
  - Changed `Examples:` to `Examples` (without colon)
  - Fixed triple quote positioning to start on new line

## Result

The documentation now includes:
- **Comprehensive API coverage**: Documentation for all 24+ functions across all modules
- **Individual function pages**: Each function has its own page (like NumPy) at `generated/mach.module.function.html`
- **Organized by category**: Functions grouped logically (beamforming, wavefront, geometry, I/O, etc.)
- **Proper NumPy-style docstring formatting**: Clean, consistent documentation format
- **Cross-references and type information**: Proper linking between related functions
- **Integration with existing Sphinx-Gallery examples**: Seamless integration with example gallery

**Function Coverage:**
- Core: `kernel.beamform`
- Wavefront: `plane`, `spherical`  
- Geometry: `spherical_to_cartesian`, `ultrasound_angles_to_cartesian`
- Experimental: `beamform`, `DLPackDevice`
- Visualization: `db`, `db_zero`, `plot_slice`, `save_debug_figures`
- I/O UFF: 6 functions for UFF file handling
- I/O PyMUST: 4 functions for PyMUST data
- I/O Utils: 4 utility functions for file operations

## Usage

Build the documentation with:
```bash
make docs
```

The API documentation will be available at `docs/_build/html/api.html` and linked from the main documentation index.

## Note

Some validation warnings remain but are non-critical and relate to missing optional sections like "See Also" and "Examples" which are suppressed in the configuration. 