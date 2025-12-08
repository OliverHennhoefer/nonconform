# Weighted Conformal Anomaly Detection

Use weighted conformal prediction to handle distribution shift in anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer, make_blobs
from nonconform import (
    Aggregation, ConformalDetector, Split, Pruning,
    logistic_weight_estimator, weighted_false_discovery_control,
    false_discovery_rate, statistical_power,
)

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
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)

# Fit on training data
detector.fit(X)

# Get weighted p-values
# The detector automatically estimates importance weights internally
p_values = detector.predict(X, raw=False)

# Apply Weighted Conformal Selection (WCS) for FDR control
discoveries = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"Weighted p-values range: {p_values.min():.4f} - {p_values.max():.4f}")
print(f"Discoveries with WCS (FDR control): {discoveries.sum()}")
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
    weight_estimator=logistic_weight_estimator(),
    seed=42
)

# Fit on original data
detector_shifted.fit(X)

# Predict on shifted data
p_values_shifted = detector_shifted.predict(X_shifted, raw=False)

# Apply WCS for FDR control
discoveries_shifted = weighted_false_discovery_control(
    result=detector_shifted.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nShifted data results:")
print(f"Weighted p-values range: {p_values_shifted.min():.4f} - {p_values_shifted.max():.4f}")
print(f"Discoveries with WCS: {discoveries_shifted.sum()}")
```

## Comparison with Standard Conformal Detection

```python
from scipy.stats import false_discovery_control

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

# Apply FDR control to standard conformal (BH procedure)
standard_disc = false_discovery_control(standard_p_values, method='bh') < 0.05

print(f"\nComparison on shifted data (with FDR control):")
print(f"Standard conformal discoveries (BH): {standard_disc.sum()}")
print(f"Weighted conformal discoveries (WCS): {discoveries_shifted.sum()}")
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
    weight_estimator=logistic_weight_estimator(),
    seed=42
)
weighted_detector.fit(X_train)
weighted_p_values = weighted_detector.predict(X_test_with_anomalies, raw=False)

# Apply FDR control
standard_disc_severe = false_discovery_control(standard_p_values, method='bh') < 0.05
weighted_disc_severe = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nSevere distribution shift results (with FDR control):")
print(f"Standard conformal discoveries (BH): {standard_disc_severe.sum()}")
print(f"Weighted conformal discoveries (WCS): {weighted_disc_severe.sum()}")
print(f"Empirical FDR (weighted): {false_discovery_rate(y=y_true, y_hat=weighted_disc_severe):.3f}")
print(f"Statistical Power (weighted): {statistical_power(y=y_true, y_hat=weighted_disc_severe):.3f}")
```

## Evaluation with Ground Truth

```python
# Evaluate weighted conformal selection with ground truth
# y_anomaly from breast cancer: target=0 is malignant (anomaly), target=1 is benign
y_anomaly = 1 - y

# Re-run on breast cancer data with proper FDR control
detector.fit(X)
_ = detector.predict(X, raw=False)

eval_discoveries = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nWeighted Conformal Selection Results:")
print(f"Discoveries: {eval_discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y=y_anomaly, y_hat=eval_discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_anomaly, y_hat=eval_discoveries):.3f}")
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

# Detection comparison (with FDR control)
detection_comparison = {
    'Standard (BH)': standard_disc_severe.sum(),
    'Weighted (WCS)': weighted_disc_severe.sum(),
}
axes[1, 1].bar(detection_comparison.keys(), detection_comparison.values())
axes[1, 1].set_ylabel('Number of Discoveries')
axes[1, 1].set_title('Discovery Comparison (with FDR control)')

plt.tight_layout()
plt.show()
```

## Different Aggregation Methods

```python
# Compare different aggregation methods for weighted conformal
aggregation_methods = [
    Aggregation.MEAN,
    Aggregation.MEDIAN,
    Aggregation.MAXIMUM,
]

for agg_method in aggregation_methods:
    det = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=agg_method,
        weight_estimator=logistic_weight_estimator(),
        seed=42
    )
    det.fit(X_train)
    _ = det.predict(X_test_with_anomalies, raw=False)

    disc = weighted_false_discovery_control(
        result=det.last_result,
        alpha=0.05,
        pruning=Pruning.DETERMINISTIC,
        seed=42,
    )
    print(f"{agg_method.value} aggregation: {disc.sum()} discoveries")
```

## JaB+ Strategy with Weighted Conformal

```python
from nonconform import JackknifeBootstrap

# Use JaB+ strategy for better stability
jab_strategy = JackknifeBootstrap(n_bootstraps=50)

weighted_jab_detector = ConformalDetector(
    detector=base_detector,
    strategy=jab_strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=logistic_weight_estimator(),
    seed=42
)

weighted_jab_detector.fit(X_train)
_ = weighted_jab_detector.predict(X_test_with_anomalies, raw=False)

# Apply WCS for FDR control
jab_discoveries = weighted_false_discovery_control(
    result=weighted_jab_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nJaB+ + Weighted Conformal (with WCS):")
print(f"Discoveries: {jab_discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y=y_true, y_hat=jab_discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_true, y_hat=jab_discoveries):.3f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation
