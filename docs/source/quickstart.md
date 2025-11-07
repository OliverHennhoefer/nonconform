# Quickstart Guide

This guide will get you started with `nonconform` in just a few minutes.

## Built-in Datasets

For quick experimentation, nonconform includes several benchmark anomaly detection datasets. Install with `pip install nonconform[data]` to enable dataset functionality.

```python
from nonconform.utils.data import load, Dataset

# Load a dataset - automatically downloads and caches
x_train, x_test, y_test = load(Dataset.BREAST, setup=True)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Anomaly ratio in test set: {y_test.mean():.2%}")
```

**Note**: Datasets are downloaded on first use and cached both in memory and on disk for faster subsequent loads.

Available datasets: Use `load(Dataset.DATASET_NAME)` where DATASET_NAME can be `BREAST`, `FRAUD`, `IONOSPHERE`, `MAMMOGRAPHY`, `MUSK`, `SHUTTLE`, `THYROID`, `WBC`.

## Basic Usage

### 1. Classical Conformal Anomaly Detection

The most straightforward way to use nonconform is with classical conformal anomaly detection:

```python
import numpy as np
from pyod.models.iforest import IForest
from sklearn.datasets import make_blobs
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation

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

print(f"P-values range: {p_values.min():.4f} - {p_values.max():.4f}")
print(f"Number of potential anomalies (p < 0.05): {(p_values < 0.05).sum()}")
```

### 2. False Discovery Rate Control

Control the False Discovery Rate using scipy's Benjamini-Hochberg procedure:

```python
from scipy.stats import false_discovery_control

# Control FDR at 5% level
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

print(f"Number of discoveries: {discoveries.sum()}")
print(f"Adjusted p-values range: {adjusted_p_values.min():.4f} - {adjusted_p_values.max():.4f}")

# Get indices of discovered anomalies
anomaly_indices = np.where(discoveries)[0]
print(f"Discovered anomaly indices: {anomaly_indices}")
```

### 3. Resampling-based Strategies

For better performance in low-data regimes, use resampling-based strategies:

```python
from nonconform.strategy import Jackknife, CrossValidation

# Jackknife (Leave-One-Out) Conformal Anomaly Detection
jackknife_strategy = Jackknife()
jackknife_detector = ConformalDetector(
    detector=base_detector,
    strategy=jackknife_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
jackknife_detector.fit(X_normal)
jackknife_p_values = jackknife_detector.predict(X_test, raw=False)

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

print("Comparison of strategies:")
print(f"Split: {(p_values < 0.05).sum()} detections")
print(f"Jackknife: {(jackknife_p_values < 0.05).sum()} detections")
print(f"Cross-Validation: {(cv_p_values < 0.05).sum()} detections")
```

## Weighted Conformal p-values

When dealing with covariate shift, use weighted conformal p-values:

```python
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import Split

# Create weighted conformal anomaly detector
weighted_strategy = Split(n_calib=0.3)
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=weighted_strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)
weighted_detector.fit(X_normal)

# Get weighted p-values
# The detector automatically estimates importance weights internally
weighted_p_values = weighted_detector.predict(X_test, raw=False)

print(f"Weighted p-values range: {weighted_p_values.min():.4f} - {weighted_p_values.max():.4f}")

# Optionally apply Weighted Conformal Selection for FDR control
from nonconform.utils.stat import weighted_false_discovery_control

selected = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

# detector.last_result bundles the cached scores and weights for reuse

print(f"Weighted FDR-controlled detections: {selected.sum()}")
```

## Integration with PyOD

nonconform integrates seamlessly with PyOD detectors:

```python
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from nonconform.strategy import Split

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
    detections = (p_vals < 0.05).sum()
    results[name] = detections
    print(f"{name}: {detections} detections")
```

## Complete Example

Here's a complete example that ties everything together:

```python
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from sklearn.datasets import make_blobs
from scipy.stats import false_discovery_control
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation

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

# Get p-values and control FDR
p_values = detector.predict(X_test, raw=False)
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

# Evaluate results
true_positives = np.sum(discoveries & (y_true == 1))
false_positives = np.sum(discoveries & (y_true == 0))
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / np.sum(y_true == 1)

print(f"Results with FDR control at 5%:")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Empirical FDR: {false_positives / max(1, discoveries.sum()):.3f}")
```

## Next Steps

- Read the [User Guide](user_guide/conformal_inference.md) for detailed explanations
- Check out the [Examples](examples/index.md) for more complex use cases
- Explore the [API Reference](api/index.md) for detailed documentation
