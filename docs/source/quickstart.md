# Quickstart Guide

Get started with nonconform in minutes.

## Benchmark Datasets (via oddball)

For quick experimentation, use `oddball`, which provides benchmark anomaly detection datasets. Install it via `pip install oddball` or `pip install "nonconform[data]"` to pull it in as an optional extra.

```python
from oddball import Dataset, load

# Load a dataset - automatically downloads and caches through oddball
x_train, x_test, y_test = load(Dataset.BREASTW, setup=True)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Anomaly ratio in test set: {y_test.mean():.2%}")
```

!!! info "Dataset Caching"
    Datasets download on first use and cache both in memory and on disk for faster subsequent loads.

Available datasets: Use `load(Dataset.DATASET_NAME)` where DATASET_NAME can be `BREASTW`, `FRAUD`, `IONOSPHERE`, `MAMMOGRAPHY`, `MUSK`, `SHUTTLE`, `THYROID`, `WBC`, and more (see `oddball.list_available()`).

## Basic Usage

### 1. Classical Conformal Anomaly Detection

Classical conformal anomaly detection:

```python
import numpy as np
from pyod.models.iforest import IForest
from sklearn.datasets import make_blobs
from scipy.stats import false_discovery_control
from nonconform import Aggregation, ConformalDetector, Split, false_discovery_rate, statistical_power

# Generate some example data
X_normal, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
X_test, _ = make_blobs(n_samples=100, centers=1, random_state=123)

# Add some anomalies to test set
X_anomalies = np.random.uniform(-10, 10, (20, X_test.shape[1]))
X_test = np.vstack([X_test, X_anomalies])

# Initialize base detector
base_detector = IForest(behaviour="new", random_state=42)

# Create conformal anomaly detector with split strategy
strategy = Split(n_calib=0.3)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit on normal data
detector.fit(X_normal)

# Get p-values for test instances
p_values = detector.predict(X_test, raw=False)

# Apply FDR control (Benjamini-Hochberg)
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

print(f"P-values range: {p_values.min():.4f} - {p_values.max():.4f}")
print(f"Discoveries with FDR control: {discoveries.sum()}")

# Get indices of discovered anomalies
anomaly_indices = np.where(discoveries)[0]
print(f"Discovered anomaly indices: {anomaly_indices}")
```

### 3. Resampling-based Strategies

For small datasets, use resampling-based strategies:

```python
from nonconform import CrossValidation, JackknifeBootstrap

# Cross-Validation Conformal Anomaly Detection
cv_strategy = CrossValidation(k=5)
cv_detector = ConformalDetector(
    detector=base_detector,
    strategy=cv_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
cv_detector.fit(X_normal)
cv_p_values = cv_detector.predict(X_test, raw=False)

# Jackknife+-after-Bootstrap (JaB+) Strategy
jab_strategy = JackknifeBootstrap(n_bootstraps=50)
jab_detector = ConformalDetector(
    detector=base_detector,
    strategy=jab_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
jab_detector.fit(X_normal)
jab_p_values = jab_detector.predict(X_test, raw=False)

# Apply FDR control to all strategies for fair comparison
split_discoveries = false_discovery_control(p_values, method='bh') < 0.05
cv_discoveries = false_discovery_control(cv_p_values, method='bh') < 0.05
jab_discoveries = false_discovery_control(jab_p_values, method='bh') < 0.05

print("Comparison of strategies (with FDR control):")
print(f"Split: {split_discoveries.sum()} discoveries")
print(f"Cross-Validation: {cv_discoveries.sum()} discoveries")
print(f"JaB+: {jab_discoveries.sum()} discoveries")
```

## Weighted Conformal p-values

For covariate shift, use weighted conformal p-values:

```python
from nonconform import (
    ConformalDetector,
    Pruning,
    Split,
    logistic_weight_estimator,
    weighted_false_discovery_control,
)

# Create weighted conformal anomaly detector
weighted_strategy = Split(n_calib=0.3)
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=weighted_strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=logistic_weight_estimator(),  # (1)
    seed=42
)
weighted_detector.fit(X_normal)

# Get weighted p-values
# The detector automatically estimates importance weights internally
weighted_p_values = weighted_detector.predict(X_test, raw=False)

print(f"Weighted p-values range: {weighted_p_values.min():.4f} - {weighted_p_values.max():.4f}")

# Apply Weighted Conformal Selection for FDR control
selected = weighted_false_discovery_control(
    result=weighted_detector.last_result,  # (2)
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"Weighted FDR-controlled detections: {selected.sum()}")
```

1. Factory function returns a configured weight estimator
2. `last_result` bundles cached scores and weights for downstream analysis

!!! warning "Covariate Shift Assumption"
    Weighted conformal methods assume P(Y|X) remains stable while only P(X) changes. Ensure density ratios can be reliably estimated.

## Using Different Detectors

### PyOD Detectors

nonconform works with any PyOD detector (install with `pip install "nonconform[pyod]"`):

```python
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from nonconform import Aggregation, ConformalDetector, Split

# Try different PyOD detectors
detectors = {
    'KNN': KNN(),
    'LOF': LOF(),
    'OCSVM': OCSVM()
}

strategy = Split(n_calib=0.3)
results = {}

for name, base_det in detectors.items():
    detector = ConformalDetector(
        detector=base_det,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    detector.fit(X_normal)
    p_vals = detector.predict(X_test, raw=False)
    # Apply FDR control before counting discoveries
    disc = false_discovery_control(p_vals, method='bh') < 0.05
    results[name] = disc.sum()
    print(f"{name}: {disc.sum()} discoveries")
```

### Custom Detectors

Any detector implementing `AnomalyDetector` works. See [Detector Compatibility](user_guide/detector_compatibility.md) for custom implementations.

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from sklearn.datasets import make_blobs
from scipy.stats import false_discovery_control
from nonconform import Aggregation, ConformalDetector, Split, false_discovery_rate, statistical_power

# Generate data
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=500, centers=1, cluster_std=1.0, random_state=42)
X_test_normal, _ = make_blobs(n_samples=80, centers=1, cluster_std=1.0, random_state=123)
X_test_anomalies = np.random.uniform(-6, 6, (20, 2))
X_test = np.vstack([X_test_normal, X_test_anomalies])

# True labels (0 = normal, 1 = anomaly)
y_true = np.hstack([np.zeros(80), np.ones(20)])

# Setup and fit detector
base_detector = IForest(behaviour="new", random_state=42)
strategy = Split(n_calib=0.3)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
detector.fit(X_normal)

# Get p-values and apply FDR control
p_values = detector.predict(X_test, raw=False)
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

# Evaluate results using nonconform metrics
print(f"Results with FDR control at 5%:")
print(f"Discoveries: {discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y=y_true, y_hat=discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_true, y_hat=discoveries):.3f}")
```

## Next Steps

- Read the [User Guide](user_guide/conformal_inference.md) for detailed explanations
- Check out the [Examples](examples/index.md) for more complex use cases
- Explore the [API Reference](api/index.md) for detailed documentation
