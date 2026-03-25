# Detector Compatibility Guide

Most detectors implementing `AnomalyDetector` work with nonconform: PyOD,
scikit-learn, or custom implementations.

For strict inductive conformal/FDR guarantees, the detector must use a fixed
training-only score map after `fit(...)`.

## AnomalyDetector Protocol

Your detector must implement these four methods:

```python
from typing import Any, Self
import numpy as np

class MyDetector:
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train on normal data. Return self."""
        ...

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more anomalous."""
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return detector parameters as dict."""
        ...

    def set_params(self, **params: Any) -> Self:
        """Set parameters. Return self."""
        ...
```

The detector must also be copyable (`copy.copy` and `copy.deepcopy`).

## PyOD Detectors

PyOD detectors are supported with strict-inductive caveats. Install PyOD with:

```sh
pip install nonconform[pyod]
```

### Compatible Detectors

One-class classification detectors work with nonconform:

| Detector | Class | Best For |
|----------|-------|----------|
| Isolation Forest | `IForest` | High-dimensional data, large datasets |
| Local Outlier Factor | `LOF` | Dense clusters, local anomalies |
| K-Nearest Neighbors | `KNN` | Simple distance-based detection |
| One-Class SVM | `OCSVM` | Complex boundaries, small datasets |
| PCA | `PCA` | Linear anomalies, interpretability |
| INNE | `INNE` | Isolation-based nearest-neighbor ensembles |
| HBOS | `HBOS` | Feature independence assumptions |
| GMM | `GMM` | Probabilistic modeling |
| AutoEncoder | `AutoEncoder` | Deep learning, complex patterns |

### Strict-Inductive Unsafe (Hard-Blocked)

These detectors are blocked in `ConformalDetector` because they do not keep a
fixed training-only score rule at inference:

- `CD`
- `COF`
- `COPOD`
- `ECOD`
- `LMDD`
- `LOCI`
- `RGraph`
- `SOD`
- `SOS`

### Meta / Inherited-Risk Detectors

These are not hard-blocked, but they inherit base-detector validity risks and
require careful curation:

- `FeatureBagging`
- `LSCP`
- `SUOD` (unsafe by default if it includes blocked base detectors)

### Basic Usage

```python
from pyod.models.iforest import IForest
from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=IForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42
)
detector.fit(X_train)
p_values = detector.compute_p_values(X_test)
```

For strict inductive workflows, choose detectors that score against frozen
training state (for example `IForest`, `KNN`, `LOF`, `HBOS`, `PCA`, `OCSVM`,
`INNE`).

### Automatic Configuration

nonconform automatically adjusts PyOD detector parameters for one-class classification:

- `contamination` -> set to minimal value
- `n_jobs` -> set to `-1` (use all cores)
- `random_state` -> set to provided `seed`

## Custom Detectors

Implement the protocol to use any anomaly detection algorithm:

```python
from typing import Any, Self
import numpy as np

class MahalanobisDetector:
    """Simple Mahalanobis distance-based anomaly detector."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state
        self._mean = None
        self._cov_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self._mean = np.mean(X, axis=0)
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        self._cov_inv = np.linalg.inv(cov)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        diff = X - self._mean
        return np.sqrt(np.sum(diff @ self._cov_inv * diff, axis=1))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"random_state": self.random_state}

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

Use it like any other detector:

```python
from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=MahalanobisDetector(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="higher_is_anomalous",  # explicit for clarity; also the default
    seed=42
)
```

See `examples/custom/centroid_detector.py` for a complete working example.

## Scikit-learn Detectors

Scikit-learn detectors that implement `fit`, `decision_function`, `get_params`, and `set_params` work directly:

```python
from sklearn.svm import OneClassSVM
from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=OneClassSVM(kernel="rbf", nu=0.05),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42
)
```

If you omit `score_polarity`, nonconform defaults to:
- `"higher_is_normal"` for known sklearn normality detectors
- `"higher_is_anomalous"` for PyOD detectors and custom detectors outside
  recognized PyOD/known-sklearn families

`score_polarity="auto"` is strict and raises for custom estimators outside
recognized PyOD/known-sklearn families.

## Troubleshooting

### Missing Methods Error

```
TypeError: Detector must implement AnomalyDetector protocol. Missing methods: decision_function
```

Your detector is missing required methods. Implement all four: `fit`, `decision_function`, `get_params`, `set_params`.

### PyOD Not Installed

```
ImportError: Detector appears to be a PyOD detector, but PyOD is not installed.
```

Install PyOD: `pip install nonconform[pyod]`

### Blocked PyOD Detector

```
ValueError: PyOD detector 'ECOD' is incompatible with strict inductive conformal/FDR workflows.
```

Use an inductive-safe detector (for example `IForest`, `KNN`, `LOF`, `HBOS`,
`PCA`, `OCSVM`, `INNE`) instead of blocked batch-adaptive detectors.

### Score Direction

nonconform computes p-values using **higher scores = more anomalous** internally.

- Use `score_polarity="higher_is_anomalous"` when your detector already follows
  that convention.
- Use `score_polarity="higher_is_normal"` when larger scores mean more normal.
- Omit `score_polarity` for convenience defaults (automatic handling for known
  sklearn normality detector families, plus custom-detector fallback to
  anomalous-higher).
- Use `score_polarity="auto"` for strict detector-family validation (raises on
  custom estimators outside recognized PyOD/known-sklearn families).

### Copyability

Your detector must support `copy.copy()` and `copy.deepcopy()`. Most Python classes work by default, but if you have complex state (file handles, connections), implement `__copy__` and `__deepcopy__`.
