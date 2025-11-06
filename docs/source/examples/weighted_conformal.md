# Weighted Conformal Anomaly Detection

This example demonstrates how to use weighted conformal prediction for handling distribution shift in anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer, make_blobs
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation

# Load example data
data = load_breast_cancer()
X = data.data
y = data.target
```

## Basic Usage

```python
# Initialize base detector
base_detector = LOF()

# Create weighted conformal detector
strategy = Split(n_calib=0.2)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42,
)

# Fit on training data
detector.fit(X)

# Get weighted p-values
# The detector automatically estimates importance weights internally
p_values = detector.predict(X, raw=False)

# Get raw scores
scores = detector.predict(X, raw=True)

print(f"Weighted p-values range: {p_values.min():.4f} - {p_values.max():.4f}")
print(f"Number of anomalies detected: {(p_values < 0.05).sum()}")
```

## Handling Distribution Shift

```python
# Simulate distribution shift by adding noise
np.random.seed(42)
X_shifted = X + np.random.normal(0, 0.1, X.shape)

# Create a new detector for shifted data
detector_shifted = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)

# Fit on original data
detector_shifted.fit(X)

# Predict on shifted data
p_values_shifted = detector_shifted.predict(X_shifted, raw=False)

print(f"\nShifted data results:")
print(f"Weighted p-values range: {p_values_shifted.min():.4f} - {p_values_shifted.max():.4f}")
print(f"Number of anomalies detected: {(p_values_shifted < 0.05).sum()}")
```

## Comparison with Standard Conformal Detection

```python
# Standard conformal detector for comparison
standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit on training data
standard_detector.fit(X)

# Compare on shifted data
standard_p_values = standard_detector.predict(X_shifted, raw=False)

print(f"\nComparison on shifted data:")
print(f"Standard conformal detections: {(standard_p_values < 0.05).sum()}")
print(f"Weighted conformal detections: {(p_values_shifted < 0.05).sum()}")
print(f"Difference: {(p_values_shifted < 0.05).sum() - (standard_p_values < 0.05).sum()}")
```

## Severe Distribution Shift Example

```python
# Create training data from one distribution
X_train, _ = make_blobs(n_samples=500, centers=1, cluster_std=1.0,
                        center_box=(0.0, 1.0), random_state=42)

# Create test data from a shifted distribution
X_test, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0,
                       center_box=(2.0, 3.0), random_state=123)

# Add some anomalies to test set
X_anomalies = np.random.uniform(-3, 6, (50, X_test.shape[1]))
X_test_with_anomalies = np.vstack([X_test, X_anomalies])

# True labels for evaluation
y_true = np.hstack([np.zeros(len(X_test)), np.ones(len(X_anomalies))])

# Standard conformal detector
standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
standard_detector.fit(X_train)
standard_p_values = standard_detector.predict(X_test_with_anomalies, raw=False)

# Weighted conformal detector
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)
weighted_detector.fit(X_train)
weighted_p_values = weighted_detector.predict(X_test_with_anomalies, raw=False)

print(f"\nSevere distribution shift results:")
print(f"Standard conformal detections: {(standard_p_values < 0.05).sum()}")
print(f"Weighted conformal detections: {(weighted_p_values < 0.05).sum()}")
```

## FDR Control with Weighted Conformal

```python
# Apply WCS to weighted scores
from nonconform.utils.stat import weighted_false_discovery_control

weighted_p_values = detector.predict(X, raw=False)

discoveries = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)
# detector.last_result stores the cached conformal statistics for reuse

# Evaluate performance
true_positives = np.sum(discoveries & (y_true == 1))
false_positives = np.sum(discoveries & (y_true == 0))
precision = true_positives / max(1, discoveries.sum())
recall = true_positives / np.sum(y_true == 1)

print(f"\nWeighted Conformal + FDR Control Results:")
print(f"Discoveries: {discoveries.sum()}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Empirical FDR: {false_positives / max(1, discoveries.sum()):.3f}")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Visualize the distribution shift and detection results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Training data
axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], alpha=0.6, c='blue', s=20)
axes[0, 0].set_title('Training Data')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# Test data with anomalies
colors = ['green' if label == 0 else 'red' for label in y_true]
axes[0, 1].scatter(X_test_with_anomalies[:, 0], X_test_with_anomalies[:, 1],
                   alpha=0.6, c=colors, s=20)
axes[0, 1].set_title('Test Data (Green=Normal, Red=Anomaly)')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')

# P-value comparison
axes[1, 0].hist(standard_p_values, bins=30, alpha=0.7, label='Standard', color='blue')
axes[1, 0].hist(weighted_p_values, bins=30, alpha=0.7, label='Weighted', color='orange')
axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
axes[1, 0].set_xlabel('p-value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('P-value Distributions')
axes[1, 0].legend()

# Detection comparison
detection_comparison = {
    'Standard': (standard_p_values < 0.05).sum(),
    'Weighted': (weighted_p_values < 0.05).sum(),
    'FDR-controlled': discoveries.sum()
}
axes[1, 1].bar(detection_comparison.keys(), detection_comparison.values())
axes[1, 1].set_ylabel('Number of Detections')
axes[1, 1].set_title('Detection Comparison')

plt.tight_layout()
plt.show()
```

## Different Aggregation Methods

```python
# Compare different aggregation methods for weighted conformal
aggregation_methods = [Aggregation.MEAN, Aggregation.MEDIAN, Aggregation.MAX]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=agg_method,
        weight_estimator=LogisticWeightEstimator(seed=42),
        seed=42
    )
    detector.fit(X_train)
    p_vals = detector.predict(X_test_with_anomalies, raw=False)

    print(f"{agg_method.value} aggregation: {(p_vals < 0.05).sum()} detections")
```

## Bootstrap Strategy with Weighted Conformal

```python
from nonconform.strategy import Bootstrap

# Use bootstrap strategy for better stability
bootstrap_strategy = Bootstrap(n_bootstraps=50, resampling_ratio=0.8)

weighted_bootstrap_detector = ConformalDetector(
    detector=base_detector,
    strategy=bootstrap_strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)

weighted_bootstrap_detector.fit(X_train)
bootstrap_p_values = weighted_bootstrap_detector.predict(X_test_with_anomalies, raw=False)

print(f"\nBootstrap + Weighted Conformal:")
print(f"Detections: {(bootstrap_p_values < 0.05).sum()}")
print(f"Comparison with split strategy: {(bootstrap_p_values < 0.05).sum() - (weighted_p_values < 0.05).sum()}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation
