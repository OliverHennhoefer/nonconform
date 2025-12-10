# Weighted Conformal Anomaly Detection

Use weighted conformal prediction to handle distribution shift in anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from oddball import Dataset, load
from nonconform import (
    Aggregation, ConformalDetector, Split, Pruning,
    logistic_weight_estimator, weighted_false_discovery_control,
    false_discovery_rate, statistical_power,
)

# Load benchmark data
X, X_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)
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

# Get weighted p-values for test data
# The detector automatically estimates importance weights internally
p_values = detector.predict(X_test, raw=False)

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
# SHUTTLE dataset naturally exhibits covariate shift between train/test
# X contains normal training data, X_test contains test data with anomalies

# Standard conformal detector
standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
standard_detector.fit(X)
standard_p_values = standard_detector.predict(X_test, raw=False)

# Weighted conformal detector
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=logistic_weight_estimator(),
    seed=42
)
weighted_detector.fit(X)
weighted_p_values = weighted_detector.predict(X_test, raw=False)

# Apply FDR control
standard_disc_severe = false_discovery_control(standard_p_values, method='bh') < 0.05
weighted_disc_severe = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nDistribution shift results (with FDR control):")
print(f"Standard conformal discoveries (BH): {standard_disc_severe.sum()}")
print(f"Weighted conformal discoveries (WCS): {weighted_disc_severe.sum()}")
print(f"Empirical FDR (weighted): {false_discovery_rate(y=y_test, y_hat=weighted_disc_severe):.3f}")
print(f"Statistical Power (weighted): {statistical_power(y=y_test, y_hat=weighted_disc_severe):.3f}")
```

## Evaluation with Ground Truth

```python
# Evaluate weighted conformal selection with ground truth
# y_test contains binary labels: 0=normal, 1=anomaly

# Re-run with proper FDR control
detector.fit(X)
_ = detector.predict(X_test, raw=False)

eval_discoveries = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nWeighted Conformal Selection Results:")
print(f"Discoveries: {eval_discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=eval_discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_test, y_hat=eval_discoveries):.3f}")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Visualize detection results (using first two features for plotting)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# P-value comparison
axes[0].hist(standard_p_values, bins=30, alpha=0.7, label='Standard', color='blue')
axes[0].hist(weighted_p_values, bins=30, alpha=0.7, label='Weighted', color='orange')
axes[0].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
axes[0].set_xlabel('p-value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('P-value Distributions')
axes[0].legend()

# Detection comparison (with FDR control)
detection_comparison = {
    'Standard (BH)': standard_disc_severe.sum(),
    'Weighted (WCS)': weighted_disc_severe.sum(),
}
axes[1].bar(detection_comparison.keys(), detection_comparison.values())
axes[1].set_ylabel('Number of Discoveries')
axes[1].set_title('Discovery Comparison (with FDR control)')

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
    det.fit(X)
    _ = det.predict(X_test, raw=False)

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

weighted_jab_detector.fit(X)
_ = weighted_jab_detector.predict(X_test, raw=False)

# Apply WCS for FDR control
jab_discoveries = weighted_false_discovery_control(
    result=weighted_jab_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"\nJaB+ + Weighted Conformal (with WCS):")
print(f"Discoveries: {jab_discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=jab_discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_test, y_hat=jab_discoveries):.3f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation
