# Detector Compatibility Guide

Any detector implementing the `AnomalyDetector` protocol works with nonconform. This includes PyOD detectors, scikit-learn compatible detectors, and custom implementations.

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

PyOD detectors are fully supported and automatically adapted. Install PyOD with:

```sh
pip install nonconform[pyod]
```

### Compatible Detectors

All detectors suitable for one-class classification work with nonconform:

| Detector | Class | Best For |
|----------|-------|----------|
| Isolation Forest | `IForest` | High-dimensional data, large datasets |
| Local Outlier Factor | `LOF` | Dense clusters, local anomalies |
| K-Nearest Neighbors | `KNN` | Simple distance-based detection |
| One-Class SVM | `OCSVM` | Complex boundaries, small datasets |
| PCA | `PCA` | Linear anomalies, interpretability |
| ECOD | `ECOD` | Parameter-free, robust |
| COPOD | `COPOD` | Correlation-based anomalies |
| HBOS | `HBOS` | Feature independence assumptions |
| GMM | `GMM` | Probabilistic modeling |
| AutoEncoder | `AutoEncoder` | Deep learning, complex patterns |

### Restricted Detectors

These detectors require anomaly labels during training and are **not supported**:

- `CBLOF` - Requires clustering labels
- `COF` - Needs connectivity information
- `RGraph` - Requires graph structure
- `Sampling` - Needs anomaly examples
- `SOS` - Requires pre-computed outlier probabilities

### Basic Usage

```python
from pyod.models.iforest import IForest
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split

detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(n_calib=0.3),
    seed=42
)
detector.fit(X_train)
p_values = detector.predict(X_test)
```

### Automatic Configuration

Nonconform automatically adjusts PyOD detector parameters for one-class classification:

- `contamination` → set to minimal value
- `n_jobs` → set to `-1` (use all cores)
- `random_state` → set to provided `seed`

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
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split

detector = ConformalDetector(
    detector=MahalanobisDetector(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42
)
```

See `examples/custom_detector.py` for a complete working example.

## Scikit-learn Detectors

Scikit-learn detectors that implement `fit`, `decision_function`, `get_params`, and `set_params` work directly:

```python
from sklearn.svm import OneClassSVM
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split

detector = ConformalDetector(
    detector=OneClassSVM(kernel="rbf", nu=0.05),
    strategy=Split(n_calib=0.3),
    seed=42
)
```

> **Note:** Ensure `decision_function` returns scores where higher values indicate more anomalous samples. Some sklearn detectors return the opposite convention.

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

### Score Direction

Ensure your `decision_function` returns scores where **higher values = more anomalous**. This is the convention nonconform expects for computing p-values.

### Copyability

Your detector must support `copy.copy()` and `copy.deepcopy()`. Most Python classes work by default, but if you have complex state (file handles, connections), implement `__copy__` and `__deepcopy__`.
