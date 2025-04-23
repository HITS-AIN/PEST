[![Build Status](https://github.com/HITS-AIN/PEST/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/HITS-AIN/PEST/actions/workflows/python-package.yml?branch=main)
[![Documentation Status](https://readthedocs.org/projects/spherinator/badge/?version=latest)](https://spherinator.readthedocs.io/en/latest/?badge=latest)
![versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)


# Preprocessing Engine for Spherinator Training (PEST)

PEST preprocess simulation data and generate training data for
[Spherinator](https://github.com/HITS-AIN/Spherinator) and
[HiPSter](https://github.com/HITS-AIN/HiPSter), including arbitrary single- and multi-channel
images, 3D PPP and PPV cubes, and point clouds.

<!-- It is currently designed to work with IllustrisTNG data, either through the web-based API or locally
using downloaded snapshots. -->

<p align="center">
  <img src="logo.png" width="100" height="100">
</p>


## Installation

```bash
pip install git+https://github.com/HITS-AIN/PEST
```

## Documentation

The `PEST` documentation is part of the Spherinator documentation and can be found at:

[Read The Docs](https://spherinator.readthedocs.io/en/latest/pest.html)


## Acknowledgments

Funded by the European Union. This work has received funding from the European High-Performance Computing Joint Undertaking (JU) and Belgium, Czech Republic, France, Germany, Greece, Italy, Norway, and Spain under grant agreement No 101093441.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European High Performance Computing Joint Undertaking (JU) and Belgium, Czech Republic, France, Germany, Greece, Italy, Norway, and Spain. Neither the European Union nor the granting authority can be held responsible for them.


## License

This project is licensed under the [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0).


## Citation

If you use HiPSter in your research, we provide a [citation](./CITATION.cff) to use:

```bibtex
@article{Polsterer_Spherinator_and_HiPSter_2024,
author = {Polsterer, Kai Lars and Doser, Bernd and Fehlner, Andreas and Trujillo-Gomez, Sebastian},
title = {{Spherinator and HiPSter: Representation Learning for Unbiased Knowledge Discovery from Simulations}},
url = {https://arxiv.org/abs/2406.03810},
doi = {10.48550/arXiv.2406.03810},
year = {2024}
}
```
