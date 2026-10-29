# Third-party notices

TransEDOS is licensed under Apache-2.0. The libraries and externally loaded
assets below are separate works and remain governed by their own licence
terms. No third-party component is relicensed by the TransEDOS `LICENSE` file.

This is a first-pass inventory of direct dependencies declared by the current
source. The licence files and notices distributed by each dependency are
authoritative and must be retained when that dependency is redistributed,
particularly in a standalone or container image.

## Python dependencies

| Component | Declared version | Licence | Upstream |
| --- | --- | --- | --- |
| Flask | 3.1.3 | BSD-3-Clause | https://github.com/pallets/flask |
| pandas | 2.2.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| NumPy | 2.2.6 | BSD-3-Clause | https://github.com/numpy/numpy |
| PyTorch | 2.4.1 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| BoTorch | 0.17.2 | MIT | https://github.com/meta-pytorch/botorch |
| GPyTorch | 1.15.2 | MIT | https://github.com/cornellius-gp/gpytorch |
| pyDOE | 1.5.0 | BSD-3-Clause | https://github.com/pydoe/pydoe |
| scikit-learn | 1.6.1 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| Matplotlib | 3.9.4 | Matplotlib License (PSF-compatible) | https://github.com/matplotlib/matplotlib |
| SciPy | 1.15.3 | BSD-3-Clause | https://github.com/scipy/scipy |
| Werkzeug | 3.1.6 | BSD-3-Clause | https://github.com/pallets/werkzeug |
| Gunicorn | 23.0.0 | MIT | https://github.com/benoitc/gunicorn |
| SHAP | >=0.40.0 | MIT | https://github.com/shap/shap |
| RDKit | 2026.3.6 | BSD-3-Clause | https://github.com/rdkit/rdkit |
| python-dotenv | >=1.0.0 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |

These packages may install transitive dependencies with additional licence
terms. Their installed metadata and licence files remain part of their
respective distributions.

## Externally loaded web resources

These resources are requested from third-party services at runtime and are not
vendored in this repository:

| Component | Use | Licence | Upstream |
| --- | --- | --- | --- |
| Plotly.js | Interactive plots | MIT | https://github.com/plotly/plotly.js |
| Inter | Web font | SIL Open Font License 1.1 | https://github.com/rsms/inter |
| GoatCounter | Optional analytics script | ISC | https://github.com/arp242/goatcounter |

The applicable licence terms should be reviewed again whenever dependencies or
packaging change.
