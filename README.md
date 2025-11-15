# mach

[![PyPI](https://img.shields.io/pypi/v/mach-beamform.svg)](https://pypi.org/project/mach-beamform/)
[![Python](https://img.shields.io/pypi/pyversions/mach-beamform.svg)](https://pypi.org/project/mach-beamform/)
[![License](https://img.shields.io/github/license/Forest-Neurotech/mach.svg)](https://github.com/Forest-Neurotech/mach/blob/main/LICENSE)
[![Actions status](https://github.com/Forest-Neurotech/mach/actions/workflows/test_gpu.yml/badge.svg)](https://github.com/Forest-Neurotech/mach/actions/)

An ultrafast CUDA-accelerated ultrasound beamformer for Python users. Developed at [Forest Neurotech](https://forestneurotech.org/).

![Benchmark Results](assets/benchmark-doppler_disk.svg)

_[Benchmark](https://github.com/Forest-Neurotech/mach/blob/main/BENCHMARKS.md): Beamforming PyMUST's [rotating-disk Doppler dataset](https://github.com/creatis-ULTIM/PyMUST/blob/170ba68/examples/rotatingDisk_real.ipynb) at 1.1 trillion points per second ([**6.5**x the speed of sound](https://github.com/Forest-Neurotech/mach/blob/main/BENCHMARKS.md))._

## Highlights

* ⚡ **Ultra-fast beamforming**: ~10x faster than prior state-of-the-art
* 🚀 **GPU-accelerated**: Leverages CUDA for maximum performance on NVIDIA GPUs
* 🎯 **Optimized for research**: Designed for functional ultrasound imaging (fUSI) and other ultrafast, high-channel-count, or volumetric-ensemble imaging
* 🐍 **Python bindings**: Zero-copy integration with CuPy, and JAX arrays via [nanobind](https://nanobind.readthedocs.io/en/latest/index.html). NumPy support included.
* 🔬 **Validated**: Matches [vbeam](https://github.com/magnusdk/vbeam) and [PyMUST](https://github.com/creatis-ULTIM/PyMUST) [outputs](https://github.com/Forest-Neurotech/mach/tree/812062f/tests/compare)


## Installation

### Install from PyPI (recommended):

```bash
pip install mach-beamform
```

Or: to include all optional dependencies, including to run the examples:
```bash
pip install mach-beamform[all]
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

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/Forest-Neurotech/mach/blob/812062f/CONTRIBUTING.md) for guidelines.

## Roadmap

### Beta release (v0.Y.0)
- ✅ Single-wave transmissions (plane wave, focused, diverging)
- ✅ Linear interpolation beamforming
- ✅ Allow NumPy/CuPy/JAX/PyTorch inputs through Array API
- ✅ Comprehensive error handling
- ✅ PyPI packaging and distribution
- ✅ Interpolation options: nearest, linear, and quadratic

### Numerically validated, but looking for feedback on API
- ✅ Coherent compounding

See the [project page](https://github.com/orgs/Forest-Neurotech/projects/14) for our up-to-date roadmap.
We welcome [feature requests](https://github.com/Forest-Neurotech/mach/issues)!

## Acknowledgments

mach builds upon the excellent work of the ultrasound imaging community:

- **[vbeam](https://github.com/magnusdk/vbeam)** - For educational examples and validation benchmarks
- **[PyMUST](https://github.com/creatis-ULTIM/PyMUST) / [PICMUS](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/)** - For standardized evaluation datasets
- **Community contributors** - Gev and Qi for CUDA optimization guidance

This package was developed by the [Forest Neurotech](https://forestneurotech.org/) team, a [Focused Research Organization](https://www.convergentresearch.org/about-fros) supported by [Convergent Research](https://www.convergentresearch.org/) and [generous philanthropic funders](https://www.convergentresearch.org/fro-portfolio).

## Citation

If you use mach in your research, you can cite:

```bibtex
@inproceedings{mach,
  title={{Mach: Beamforming one trillion points per second on a consumer GPU}},
  author={Guan, Charles and Rockhill, Alex and Pinton, Gianmarco},
  booktitle={Medical Imaging 2026: Ultrasonic Imaging and Tomography},
  year={2026},
  organization = {International Society for Optics and Photonics},
  publisher = {SPIE},
  URL={https://github.com/Forest-Neurotech/mach}
}
```
