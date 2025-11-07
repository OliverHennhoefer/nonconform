![Logo](https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/banner_light.png)

---

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
[![codecov](https://codecov.io/gh/OliverHennhoefer/nonconform/graph/badge.svg?token=Z78HU3I26P)](https://codecov.io/gh/OliverHennhoefer/nonconform)

  ## Conformal Anomaly Detection
Thresholds for anomaly detection are often arbitrary and lack theoretical guarantees about the anomalies they identify. **nonconform** wraps your favorite anomaly detection model from [*PyOD*](https://pyod.readthedocs.io/en/latest/) (see [Supported Estimators](#supported-estimators)) and transforms its raw anomaly scores into statistically valid $p$-values. It applies principles from [**conformal prediction**](https://en.wikipedia.org/wiki/Conformal_prediction) to the setting of [one-class classification](https://en.wikipedia.org/wiki/One-class_classification), enabling anomaly detection with provable statistical guarantees and a controlled [false discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate).

> **Note:** The methods in **nonconform** assume that training and test data are [*exchangeable*](https://en.wikipedia.org/wiki/Exchangeable_random_variables). Therefore, the package is not suited for data with spatial or temporal autocorrelation unless such dependencies are explicitly handled in preprocessing or model design.


# :hatching_chick: Getting Started

Installation via [PyPI](https://pypi.org/project/nonconform/):
```sh
pip install nonconform
```

> **Note:** The following examples use the built-in datasets. Install with `pip install nonconform[data]` to run these examples. (see [Optional Dependencies](#optional-dependencies))


## Classical (Conformal) Approach

**Example:** Detecting anomalies with Isolation Forest on the Shuttle dataset. The approach splits data for calibration, trains the model, then converts anomaly scores to p-values by comparing test scores against the calibration distribution.

```python
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform.strategy import Split
from nonconform.detection import ConformalDetector
from nonconform.utils.data import load, Dataset
from nonconform.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

estimator = ConformalDetector(
 detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000), seed=42)

estimator.fit(x_train)

estimates = estimator.predict(x_test)
decisions = false_discovery_control(estimates, method='bh') <= 0.2

print(f"Empirical False Discovery Rate: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Statistical Power (Recall): {statistical_power(y=y_test, y_hat=decisions)}")
```

Output:
```text
Empirical False Discovery Rate: 0.18
Empirical Statistical Power (Recall): 0.99
```

# :hatched_chick: Advanced Methods

Two advanced approaches are implemented that may increase the power of a conformal anomaly detector:
- A KDE-based (probabilistic) approach that models the calibration scores to achieve continuous $p$-values in contrast to the standard empirical distribution function.
- A weighted approach that prioritizes calibration scores by their similarity to the test batch at hand and is more robust to covariate shift between test and calibration data. Maybe combine with the probabilistic approach.

Probabilistic Conformal Approach:

````python
estimator = ConformalDetector(
        detector=HBOS(),
        strategy=Split(n_calib=1_000),
        estimation=Probabilistic(n_trials=10),  # KDE Tuning Trials
        seed=1,
    )
````

Weighed Conformal Anomaly Detection:

```python
# Weighted conformal (with covariate shift handling):
from nonconform.detection.weight import LogisticWeightEstimator

estimator = ConformalDetector(
 detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000), weight_estimator=LogisticWeightEstimator(seed=42), seed=42)
```

> **Note:** Weighted procedures require weighted FDR control for statistical validity (see ``weighted_bh()`` or ``weighted_false_discovery_control()``).


# Beyond Static Data

While primarily designed for static (single-batch) applications, the library supports streaming scenarios through ``BatchGenerator()`` and ``OnlineGenerator()``. For statistically valid FDR control in streaming data, use the optional ``onlineFDR`` dependency, which implements appropriate statistical methods.


# Citation

If you find this repository useful for your research, please cite the following papers:

##### Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors
```text
@inproceedings{Hennhofer2024,
 title        = {{ Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors }}, author       = {Hennhofer, Oliver and Preisach, Christine}, year         = 2024, month        = {Dec}, booktitle    = {2024 IEEE International Conference on Knowledge Graph (ICKG)}, publisher    = {IEEE Computer Society}, address      = {Los Alamitos, CA, USA}, pages        = {110--119}, doi          = {10.1109/ICKG63256.2024.00022}, url          = {https://doi.ieeecomputersociety.org/10.1109/ICKG63256.2024.00022}}
```

##### Testing for Outliers with Conformal p-Values
```text
@article{Bates2023,
 title        = {Testing for outliers with conformal p-values}, author       = {Bates,  Stephen and Candès,  Emmanuel and Lei,  Lihua and Romano,  Yaniv and Sesia,  Matteo}, year         = 2023, month        = feb, journal      = {The Annals of Statistics}, publisher    = {Institute of Mathematical Statistics}, volume       = 51, number       = 1, doi          = {10.1214/22-aos2244}, issn         = {0090-5364}, url          = {http://dx.doi.org/10.1214/22-AOS2244}}
```

# Optional Dependencies

_For additional features, you might need optional dependencies:_
- `pip install nonconform[data]` - Includes pyarrow for loading example data (via remote download)
- `pip install nonconform[deep]` - Includes deep learning dependencies (PyTorch)
- `pip install nonconform[fdr]` - Includes advanced FDR control methods (online-fdr)
- `pip install nonconform[dev]` - Includes development tools documentation tools
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

---
