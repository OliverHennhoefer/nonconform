<!-- start-github-only -->
![Logo](./docs/img/banner_dark.png#gh-dark-mode-only)
![Logo](./docs/img/banner_light.png#gh-light-mode-only)
<!-- end-github-only -->
<!-- start-pypi-only -->
![Logo](https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/banner_light.png)
<!-- end-pypi-only -->

---

![License](https://img.shields.io/github/license/OliverHennhoefer/nonconform)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/OliverHennhoefer/nonconform)
![PyPI](https://img.shields.io/pypi/v/nonconform)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linting-ruff-%23FFA500)](https://github.com/astral-sh/ruff)
![Hatch](https://img.shields.io/badge/build-hatch-FF69B4)




**nonconform** enhances anomaly detection by providing uncertainty quantification. It acts as a wrapper around most detectors from [*PyOD*](https://pyod.readthedocs.io/en/latest/) (see [Supported Estimators](#supported-estimators)). By leveraging one-class classification and **conformal inference**, **nonconform** enables **statistically rigorous anomaly detection**.

*   **Uncertainty Quantification:** Turn anomaly scores into statistically valid _p_-values.
*   **Control False Positives:** Reliably control metrics like the False Discovery Rate (FDR).
*   **[*PyOD*](https://pyod.readthedocs.io/en/latest/) Compatibility:** Works with most PyOD anomaly detectors (see [Supported Estimators](#supported-estimators)).

# :hatching_chick: Getting Started

Installation via [PyPI](https://pypi.org/project/nonconform/):
```sh
pip install nonconform
```

> **Note:** The following examples use the built-in datasets. Install with `pip install nonconform[data]` to run these examples. (see [Optional Dependencies](#optional-dependencies))


## Classical (Conformal) Approach

**Example:** Detecting anomalies with Isolation Forest on the Shuttle dataset. The approach splits data for calibration, trains the model, then converts anomaly scores to statistically valid p-values by comparing test scores against the calibration distribution.

```python
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform.strategy import Split
from nonconform.estimation import StandardConformalDetector
from nonconform.utils.data import load_shuttle
from nonconform.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True, seed=42)

estimator = StandardConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(n_calib=2_000),
    seed=42
)

estimator.fit(x_train)

estimates = estimator.predict(x_test)
decisions = false_discovery_control(estimates, method='bh') <= 0.2

print(f"Empirical False Discovery Rate: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Statistical Power (Recall): {statistical_power(y=y_test, y_hat=decisions)}")
```

Output:
```text
Empirical False Discovery Rate: 0.198
Empirical Statistical Power (Recall): 0.97
```

# :hatched_chick: Advanced Methods

Other conformal detector wrappers exist for advanced use cases, including ``WeightedConformalDetector()`` (robust to covariate shifts) and sophisticated calibration strategies like ``JackknifeBootstrap()`` for improved results.


# Beyond Static Data

While primarily designed for static (single-batch) applications, the library supports streaming scenarios through ``BatchGenerator()`` and ``OnlineGenerator()``. For statistically valid FDR control in streaming data, use the optional ``onlineFDR`` dependency, which implements appropriate statistical methods.


# Citation

If you find this repository useful for your research, please cite the following papers:

##### Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors
```text
@inproceedings{Hennhofer2024,
	title        = {{ Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors }},
	author       = {Hennhofer, Oliver and Preisach, Christine},
	year         = 2024,
	month        = {Dec},
	booktitle    = {2024 IEEE International Conference on Knowledge Graph (ICKG)},
	publisher    = {IEEE Computer Society},
	address      = {Los Alamitos, CA, USA},
	pages        = {110--119},
	doi          = {10.1109/ICKG63256.2024.00022},
	url          = {https://doi.ieeecomputersociety.org/10.1109/ICKG63256.2024.00022}
}
```

##### Testing for Outliers with Conformal p-Values
```text
@article{Bates2023,
	title        = {Testing for outliers with conformal p-values},
	author       = {Bates,  Stephen and Candès,  Emmanuel and Lei,  Lihua and Romano,  Yaniv and Sesia,  Matteo},
	year         = 2023,
	month        = feb,
	journal      = {The Annals of Statistics},
	publisher    = {Institute of Mathematical Statistics},
	volume       = 51,
	number       = 1,
	doi          = {10.1214/22-aos2244},
	issn         = {0090-5364},
	url          = {http://dx.doi.org/10.1214/22-AOS2244}
}
```

# Optional Dependencies

_For additional features, you might need optional dependencies:_
- `pip install nonconform[data]` - Includes pyarrow for loading example data (via remote download)
- `pip install nonconform[deep]` - Includes deep learning dependencies (PyTorch)
- `pip install nonconform[fdr]` - Includes advanced FDR control methods (online-fdr)
- `pip install nonconform[dev]` - Includes development tools (black, ruff, pre-commit) and documentation tools (sphinx, furo, etc.)
- `pip install nonconform[all]` - Includes all optional dependencies

_Please refer to the [pyproject.toml](https://github.com/OliverHennhoefer/nonconform/blob/main/pyproject.toml) for details._

# Supported Estimators

Only anomaly estimators suitable for unsupervised one-class classification are supported. Since detectors are trained exclusively on normal data, threshold parameters are automatically set to minimal values.

Models that are **currently supported** include:

* Angle-Based Outlier Detection (**ABOD**)
* Autoencoder (**AE**)
* Cook's Distance (**CD**)
* Copula-based Outlier Detector (**COPOD**)
* Deep Isolation Forest (**DIF**)
* Empirical-Cumulative-distribution-based Outlier Detection (**ECOD**)
* Gaussian Mixture Model (**GMM**)
* Histogram-based Outlier Detection (**HBOS**)
* Isolation-based Anomaly Detection using Nearest-Neighbor Ensembles (**INNE**)
* Isolation Forest (**IForest**)
* Kernel Density Estimation (**KDE**)
* *k*-Nearest Neighbor (***k*NN**)
* Kernel Principal Component Analysis (**KPCA**)
* Linear Model Deviation-base Outlier Detection (**LMDD**)
* Local Outlier Factor (**LOF**)
* Local Correlation Integral (**LOCI**)
* Lightweight Online Detector of Anomalies (**LODA**)
* Locally Selective Combination of Parallel Outlier Ensembles (**LSCP**)
* GNN-based Anomaly Detection Method (**LUNAR**)
* Median Absolute Deviation (**MAD**)
* Minimum Covariance Determinant (**MCD**)
* One-Class SVM (**OCSVM**)
* Principal Component Analysis (**PCA**)
* Quasi-Monte Carlo Discrepancy Outlier Detection (**QMCD**)
* Rotation-based Outlier Detection (**ROD**)
* Subspace Outlier Detection (**SOD**)
* Scalable Unsupervised Outlier Detection (**SUOD**)

# Contact
**Bug reporting:** [https://github.com/OliverHennhoefer/nonconform/issues](https://github.com/OliverHennhoefer/nonconform/issues)
