![Logo](https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/banner_light.png)

---

![Python versions](https://img.shields.io/pypi/pyversions/nonconform.svg)
[![codecov](https://codecov.io/gh/OliverHennhoefer/nonconform/branch/main/graph/badge.svg?token=Z78HU3I26P)](https://codecov.io/gh/OliverHennhoefer/nonconform)
[![PyPI version](https://img.shields.io/pypi/v/nonconform.svg)](https://pypi.org/project/nonconform/)
[![Docs](https://github.com/OliverHennhoefer/nonconform/actions/workflows/docs.yml/badge.svg)](https://oliverhennhoefer.github.io/nonconform/)

## Conformal Anomaly Detection

Thresholds for anomaly detection are often arbitrary and lack theoretical guarantees. **nonconform** wraps anomaly detectors (from [PyOD](https://pyod.readthedocs.io/en/latest/), scikit-learn, or custom implementations) and transforms their raw anomaly scores into statistically valid *p*-values. It applies principles from [conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) to [one-class classification](https://en.wikipedia.org/wiki/One-class_classification), enabling anomaly detection with provable statistical guarantees and a controlled [false discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate).

> **Note:** The methods in **nonconform** assume that training and test data are [*exchangeable*](https://en.wikipedia.org/wiki/Exchangeable_random_variables) [Vovk et al., 2005]. Therefore, the package is not suited for data with spatial or temporal autocorrelation unless such dependencies are explicitly handled in preprocessing or model design.


# :hatching_chick: Getting Started

Installation via [PyPI](https://pypi.org/project/nonconform/):
```sh
pip install nonconform
```

> **Note:** The following examples use an external dataset API. Install with `pip install oddball` or `pip install "nonconform[data]"` to include it. (see [Optional Dependencies](#optional-dependencies))


## Classical (Conformal) Approach

**Example:** Detecting anomalies with Isolation Forest on the Shuttle dataset. The approach splits data for calibration, trains the model, then converts anomaly scores to p-values by comparing test scores against the calibration distribution. See [ConformalDetector](https://oliverhennhoefer.github.io/nonconform/api/#nonconform.detector.ConformalDetector), [Split](https://oliverhennhoefer.github.io/nonconform/api/#nonconform.resampling.Split), and [FDR Control](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/).

```python
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power
from oddball import Dataset, load

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(n_calib=1_000),
    seed=42,
)
p_values = detector.fit(x_train).compute_p_values(x_test)
decisions = false_discovery_control(p_values, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y_test, decisions)}")
print(f"Statistical Power: {statistical_power(y_test, decisions)}")
```

Output:
```text
Empirical FDR: 0.18
Statistical Power: 0.99
```

# :hatched_chick: Advanced Methods

Two advanced approaches are implemented that may increase the power of a conformal anomaly detector:
- A KDE-based ([probabilistic](https://oliverhennhoefer.github.io/nonconform/api/#nonconform.scoring.Probabilistic)) approach that models the calibration scores to achieve continuous *p*-values in contrast to the standard empirical distribution function.
- A [weighted](https://oliverhennhoefer.github.io/nonconform/user_guide/weighted_conformal/) approach that prioritizes calibration scores by their similarity to the test batch at hand and is more robust to covariate shift between test and calibration data (can be combined with the probabilistic approach).
- [Exchangeability martingales](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/) for sequential evidence monitoring on conformal *p*-value streams (`PowerMartingale`, `SimpleMixtureMartingale`, `SimpleJumperMartingale`).

Probabilistic Conformal Approach:

```python
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split, Probabilistic

detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(n_calib=1_000),
    estimation=Probabilistic(n_trials=10),
    seed=42,
)
```

Weighted Conformal Anomaly Detection:

```python
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(n_calib=1_000),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
```

> **Note:** Weighted procedures require weighted FDR control for statistical validity (see [`nonconform.fdr.weighted_false_discovery_control()`](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/#weighted-conformal-selection)).


Exchangeability Martingales (sequential monitoring):

This snippet shows martingale setup only. In normal use:
- a fitted `ConformalDetector` produces streaming conformal p-values from model scores
- each incoming p-value is fed to the martingale via `martingale.update(p_t)`

```python
from nonconform.martingales import AlarmConfig, PowerMartingale

martingale = PowerMartingale(
    epsilon=0.5,
    alarm_config=AlarmConfig(ville_threshold=100.0),
)

# update one p-value at a time
state = martingale.update(float(p_t))

# or update a sequence of p-values
states = martingale.update_many(p_values_chunk)
```

> **Note:** Martingale alarms are evidence-monitoring signals on sequential *p*-values. They are not a replacement for cross-hypothesis FDR control.
> See the user guide for a compact end-to-end flow:
> [Exchangeability Martingales](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/).


# Beyond Static Data

While primarily designed for static (single-batch) applications, the optional `onlinefdr` dependency provides FDR control methods appropriate for [streaming scenarios](https://oliverhennhoefer.github.io/nonconform/user_guide/streaming_evaluation/).


# Custom Detectors

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

For custom detectors, either set `score_polarity` explicitly
(`"higher_is_anomalous"` in most cases), or omit it to use the pre-release
default behavior. Use `score_polarity="auto"` only when you want strict
detector-family validation.

See [Detector Compatibility](https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/) for details and examples.


# Citation

If you find this repository useful for your research, please cite the following papers:

##### Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors
```bibtex
@inproceedings{Hennhofer2024,
    title     = {Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors},
    author    = {Hennhofer, Oliver and Preisach, Christine},
    year      = {2024},
    month     = {Dec},
    booktitle = {2024 IEEE International Conference on Knowledge Graph (ICKG)},
    publisher = {IEEE Computer Society},
    address   = {Los Alamitos, CA, USA},
    pages     = {110--119},
    doi       = {10.1109/ICKG63256.2024.00022},
    url       = {https://doi.ieeecomputersociety.org/10.1109/ICKG63256.2024.00022}
}
```

##### Testing for Outliers with Conformal p-Values
```bibtex
@article{Bates2023,
    title     = {Testing for outliers with conformal p-values},
    author    = {Bates, Stephen and Candès, Emmanuel and Lei, Lihua and Romano, Yaniv and Sesia, Matteo},
    year      = {2023},
    month     = {Feb},
    journal   = {The Annals of Statistics},
    publisher = {Institute of Mathematical Statistics},
    volume    = {51},
    number    = {1},
    doi       = {10.1214/22-aos2244},
    issn      = {0090-5364},
    url       = {http://dx.doi.org/10.1214/22-AOS2244}
}
```

##### Algorithmic Learning in a Random World
```bibtex
@book{Vovk2005,
    title     = {Algorithmic Learning in a Random World},
    author    = {Vladimir Vovk and Alex Gammerman and Glenn Shafer},
    year      = {2005},
    publisher = {Springer},
    note      = {Springer, New York},
    language  = {English}
}
```

##### Testing Exchangeability On-line
```bibtex
@inproceedings{Vovk2003,
    title     = {Testing Exchangeability On-line},
    author    = {Vovk, Vladimir and Nouretdinov, Ilia and Gammerman, Alex},
    booktitle = {Proceedings of the 20th International Conference on Machine Learning (ICML)},
    year      = {2003}
}
```

##### Retrain or Not Retrain: Conformal Test Martingales for Change-Point Detection
```bibtex
@inproceedings{Vovk2021,
    title     = {Retrain or Not Retrain: Conformal Test Martingales for Change-Point Detection},
    author    = {Vovk, Vladimir and Volkhonskiy, Daniil and Nouretdinov, Ilia and Gammerman, Alex},
    booktitle = {Proceedings of The 10th Symposium on Conformal and Probabilistic Prediction and Applications},
    series    = {PMLR},
    volume    = {152},
    pages     = {210--231},
    year      = {2021},
    url       = {https://proceedings.mlr.press/v152/vovk21b.html}
}
```

# Optional Dependencies

_For additional features, you might need optional dependencies:_
- `pip install nonconform[pyod]` - Includes PyOD anomaly detection library
- `pip install nonconform[data]` - Includes oddball for loading benchmark datasets
- `pip install nonconform[fdr]` - Includes advanced FDR control methods (online-fdr)
- `pip install nonconform[probabilistic]` - Includes KDEpy and Optuna for probabilistic estimation/tuning
- `pip install nonconform[all]` - Includes all optional dependencies

_Please refer to the [pyproject.toml](https://github.com/OliverHennhoefer/nonconform/blob/main/pyproject.toml) for details._

# Contact
**Bug reporting:** [https://github.com/OliverHennhoefer/nonconform/issues](https://github.com/OliverHennhoefer/nonconform/issues)

---
