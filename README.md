![Logo](./docs/img/banner_dark.png#gh-dark-mode-only)
![Logo](./docs/img/banner_light.png#gh-light-mode-only)

---

![Python versions](https://img.shields.io/pypi/pyversions/nonconform.svg)
[![codecov](https://codecov.io/gh/OliverHennhoefer/nonconform/branch/main/graph/badge.svg?token=Z78HU3I26P)](https://codecov.io/gh/OliverHennhoefer/nonconform)
[![PyPI version](https://img.shields.io/pypi/v/nonconform.svg)](https://pypi.org/project/nonconform/)
[![Docs](https://github.com/OliverHennhoefer/nonconform/actions/workflows/docs.yml/badge.svg)](https://oliverhennhoefer.github.io/nonconform/)

## Conformal Anomaly Detection

Thresholds for anomaly detection are often arbitrary and lack theoretical guarantees. **nonconform** wraps anomaly detectors (from [PyOD](https://pyod.readthedocs.io/en/latest/), scikit-learn, or custom implementations) and transforms their raw anomaly scores into statistically valid *p*-values. It applies principles from conformal prediction to one-class anomaly detection, enabling controlled false discovery rate (FDR) workflows with explicit statistical guarantees.

> **Note:** The methods in **nonconform** assume that training and test data are [*exchangeable*](https://en.wikipedia.org/wiki/Exchangeable_random_variables). The package is therefore not suited for spatial or temporal autocorrelation unless such dependencies are explicitly handled in preprocessing or model design.

## Feature Overview

| Need | nonconform Functionality | Start Here |
| --- | --- | --- |
| Principled anomaly decisions | `ConformalDetector.select(...)` combines conformal *p*-values with FDR-controlled selection | [FDR Control](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/) |
| Flexible calibration strategies | `Split`, `CrossValidation`, and `JackknifeBootstrap` for different data/compute tradeoffs | [Conformalization Strategies](https://oliverhennhoefer.github.io/nonconform/user_guide/conformalization_strategies/) |
| Covariate-shift aware workflows | Weighted conformal prediction with density-ratio estimators and weighted FDR control (requires sufficient calibration/test support overlap) | [Weighted Conformal](https://oliverhennhoefer.github.io/nonconform/user_guide/weighted_conformal/) |
| Labeled outlier workflows | `IntegrativeConformalDetector` for labeled inliers and labeled outliers OOD testing with split conditional-FDR selection | [Integrative Conformal](https://oliverhennhoefer.github.io/nonconform/user_guide/integrative_conformal/) |
| Rich p-value estimation | Empirical, probabilistic KDE, and conditional calibration estimators | [Common Workflows](https://oliverhennhoefer.github.io/nonconform/api/common_workflows/) |
| Sequential monitoring | Exchangeability martingales (`PowerMartingale`, `SimpleMixtureMartingale`, `SimpleJumperMartingale`) | [Exchangeability Martingales](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/) |
| Custom detector integration | Support for any detector implementing the `AnomalyDetector` protocol | [Detector Compatibility](https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/) |

## Getting Started

Installation via [PyPI](https://pypi.org/project/nonconform/):

```sh
pip install nonconform
```

> **Note:** The example below uses an external dataset API. Install with `pip install oddball` or `pip install "nonconform[data]"`.

### Classical Conformal Workflow

**Example:** Isolation Forest on the Shuttle benchmark. This trains a base detector, calibrates conformal scores, then applies FDR-controlled selection through `select(...)`. Raw *p*-values remain available via `detector.last_result.p_values`.

```python
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power
from oddball import Dataset, load

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=1_000),
    seed=42,
)
detector.fit(x_train)

decisions = detector.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y_test, decisions)}")
print(f"Statistical Power: {statistical_power(y_test, decisions)}")
```

Output:

```text
Empirical FDR: 0.18
Statistical Power: 0.99
```

## Advanced Methods

nonconform includes advanced workflows for practitioners who need more power or robustness:

- **Probabilistic Conformal Estimation** (`Probabilistic`): uses KDE-based modeling of calibration scores to produce continuous *p*-values instead of purely empirical stepwise values.
- **Weighted Conformal Prediction** (`weight_estimator=...`): reweights calibration evidence for covariate shift settings where test and calibration distributions differ, assuming enough support overlap between calibration and test features.
- **Integrative Conformal Detection** (`IntegrativeConformalDetector`): combines labeled inliers and labeled outliers in a dedicated conformal workflow instead of treating the problem as covariate shift.
- **Exchangeability Martingales** (`nonconform.martingales`): sequential evidence monitoring over conformal *p*-value streams.

Probabilistic Conformal Setup:

```python
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Probabilistic, Split

detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=1_000),
    estimation=Probabilistic(n_trials=10),
    seed=42,
)
```

Weighted Conformal Setup:

```python
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=1_000),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
```

> **Note:** In weighted mode, `ConformalDetector.select(...)` dispatches weighted FDR control automatically.

Martingale Setup for Sequential Monitoring:

```python
from nonconform.martingales import AlarmConfig, PowerMartingale

martingale = PowerMartingale(
    epsilon=0.5,
    alarm_config=AlarmConfig(ville_threshold=100.0),
)

state = martingale.update(p_t)
states = martingale.update_many(p_values_chunk)
```

> **Note:** `update(...)` already validates and normalizes numeric scalar p-values, so an explicit `float(...)` cast is optional.
> Martingale alarms monitor evidence over time; they do not replace cross-hypothesis FDR control.

## Beyond Static Data

While primarily designed for static (single-batch) workflows, optional `onlinefdr` integration supports [streaming FDR procedures](https://oliverhennhoefer.github.io/nonconform/user_guide/streaming_evaluation/).

## Custom Detectors

Any detector implementing the [`AnomalyDetector`](https://oliverhennhoefer.github.io/nonconform/api/#nonconform.structures.AnomalyDetector) protocol works with nonconform:

```python
from typing import Self

import numpy as np

class MyDetector:
    def fit(self, X, y=None) -> Self: ...
    def decision_function(self, X) -> np.ndarray: ...  # higher = more anomalous
    def get_params(self, deep=True) -> dict: ...
    def set_params(self, **params) -> Self: ...
```

For custom detectors, either set `score_polarity` explicitly (`"higher_is_anomalous"` in most cases), or omit it to use the pre-release default behavior. Use `score_polarity="auto"` only when you want strict detector-family validation.

See [Detector Compatibility](https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/) for details and examples.

## Optional Dependencies

_For additional features, you might need optional dependencies:_

- `pip install nonconform[pyod]` - Includes PyOD anomaly detection library
- `pip install nonconform[data]` - Includes oddball for loading benchmark datasets
- `pip install nonconform[fdr]` - Includes advanced FDR control methods (online-fdr)
- `pip install nonconform[probabilistic]` - Includes KDEpy and Optuna for probabilistic estimation/tuning
- `pip install nonconform[all]` - Includes all optional dependencies

_Please refer to the [pyproject.toml](https://github.com/OliverHennhoefer/nonconform/blob/main/pyproject.toml) for details._

## Contact

**Bug reporting:** [https://github.com/OliverHennhoefer/nonconform/issues](https://github.com/OliverHennhoefer/nonconform/issues)

----

<a href="https://www.dlr.de/">
  <img src="https://www.dlr.de/de/pt-lf/aktuelles/pressematerial/logos/bmwk/vorschaubild_bmwk_logo-mit-foerderzusatz_en/@@images/image-600-ea91cd9090327104991124b30fe1de7d.png" alt="BMWK logo" width="250"/>
</a>
