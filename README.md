# mach 5 beamformer

[![PyPI](https://img.shields.io/pypi/v/mach-beamform.svg)](https://pypi.org/project/mach-beamform/)
[![Python](https://img.shields.io/pypi/pyversions/mach-beamform.svg)](https://pypi.org/project/mach-beamform/)
[![License](https://img.shields.io/github/license/Forest-Neurotech/mach.svg)](https://github.com/Forest-Neurotech/mach/blob/main/LICENSE)
[![Actions status](https://github.com/Forest-Neurotech/mach/actions/workflows/test_gpu.yml/badge.svg)](https://github.com/Forest-Neurotech/mach/actions/)

An ultrafast CUDA-accelerated ultrasound beamformer for Python users. Developed at [Forest Neurotech](https://forestneurotech.org/).

![Benchmark Results](assets/benchmark-doppler_disk.svg)

_Beamforming PyMUST's [rotating-disk Doppler dataset](https://github.com/creatis-ULTIM/PyMUST/blob/170ba68/examples/rotatingDisk_real.ipynb) at 0.86 trillion points per second (**5**x the speed of sound)._

> **⚠️ Alpha Release**
>
> This library is currently under active development and is released as an alpha version. The primary goal of this release is to collect community feedback.


## Highlights

* ⚡ **Ultra-fast beamforming**: ~10x faster than prior state-of-the-art
* 🚀 **GPU-accelerated**: Leverages CUDA for maximum performance on NVIDIA GPUs
* 🎯 **Optimized for research**: Designed for functional ultrasound imaging (fUSI) and other ultrafast, high-channel-count, or volumetric-ensemble imaging
* 🐍 **Python bindings**: Zero-copy integration with CuPy, and JAX arrays via [nanobind](https://nanobind.readthedocs.io/en/latest/index.html). NumPy support included.
* 🔬 **Validated**: Matches [vbeam](https://github.com/magnusdk/vbeam) and [PyMUST](https://github.com/creatis-ULTIM/PyMUST) [outputs](tests/compare)


## Installation

### Install from PyPI (recommended):

```bash
pip install mach-beamform
```

Wheel prerequisites:
* [Linux](https://github.com/pypa/manylinux)
* CUDA-enabled GPU with driver >= 12.3, [compute-capability >= 7.5](https://developer.nvidia.com/cuda-gpus)

### Build from source

```bash
make compile
```
Build prerequisites:
* Linux
* `make`
* `uv >= 0.6.10`
* `gcc >= 8`
* `nvcc >= 11.0`

## Examples

Try our [examples](https://forest-neurotech.github.io/mach/examples/):

* [📊 Plane Wave Imaging with PICMUS Dataset](examples/plane_wave_compound.py)
* [🩸 Doppler Imaging](examples/doppler.py)

If you don't have a CUDA-enabled GPU, you can download the notebook from the [docs](https://forest-neurotech.github.io/mach/examples/) and open in Google Colab (select a GPU instance).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap

### Alpha release (v0.0.Z)
- ✅ Single-wave transmissions (plane wave, focused, diverging)
- ✅ Linear interpolation beamforming
- ✅ Allow NumPy/CuPy/JAX/PyTorch inputs through Array API
- ✅ Comprehensive error handling
- ✅ PyPI packaging and distribution

### Experimental features
- Coherent compounding

### Tentative Future Plans
- Additional interpolation methods (spline, sinc)
- Additional apodization windows

See the [project page](https://github.com/orgs/Forest-Neurotech/projects/14) for our up-to-date roadmap.
We welcome [feature requests](https://github.com/Forest-Neurotech/mach/issues)!

## Acknowledgments

mach builds upon the excellent work of the ultrasound imaging community:

* **[vbeam](https://github.com/magnusdk/vbeam)** - For educational examples and validation benchmarks
* **[PyMUST](https://github.com/creatis-ULTIM/PyMUST) / [PICMUS](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/)** - For standardized evaluation datasets
* **Community contributors** - Gev and Qi for CUDA optimization guidance

## Citation

If you use mach in your research, please cite:

```bibtex
@software{mach,
  title={mach: Ultra-fast GPU-accelerated ultrasound beamforming},
  author={Guan, Charles and Rockhill, Alex and Pinton, Gianmarco},
  organization={Forest Neurotech},
  year={2025},
  url={https://github.com/Forest-Neurotech/mach}
}
```
