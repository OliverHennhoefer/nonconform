# Classical Conformal Anomaly Detection

This example demonstrates how to use classical conformal prediction for anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from scipy.stats import false_discovery_control
from nonconform import Aggregation, ConformalDetector, Split, false_discovery_rate, statistical_power
from oddball import Dataset, load

# Load example data - downloads automatically and caches in memory
x_train, x_test, y_test = load(Dataset.BREASTW, setup=True)
print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
```

## Basic Usage

```python
# Initialize base detector
base_detector = LOF()

# Create conformal detector with split strategy
strategy = Split(n_calib=0.2)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit the detector on training data (normal samples only)
detector.fit(x_train)

# Get p-values for test data
p_values = detector.predict(x_test, raw=False)

# Get raw anomaly scores (optional)
scores = detector.predict(x_test, raw=True)

# Apply FDR control (Benjamini-Hochberg)
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

print(f"Discoveries with FDR control: {discoveries.sum()}")
print(f"True anomaly rate in test set: {y_test.mean():.2%}")
print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_test, y_hat=discoveries):.3f}")
```

## Advanced Usage with Cross-Validation

```python
from nonconform import CrossValidation

# Use cross-validation strategy for better calibration
cv_strategy = CrossValidation(k=5)
cv_detector = ConformalDetector(
    detector=base_detector,
    strategy=cv_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and predict with cross-validation
cv_detector.fit(x_train)
cv_p_values = cv_detector.predict(x_test, raw=False)

# Compare with split strategy
# Apply FDR control for fair comparison
split_fdr = false_discovery_control(p_values, method='bh')
cv_fdr = false_discovery_control(cv_p_values, method='bh')

print(f"Split strategy detections: {(split_fdr < 0.05).sum()}")
print(f"Cross-validation detections: {(cv_fdr < 0.05).sum()}")
```

## Comparing Different Aggregation Methods

```python
# Try different aggregation methods
aggregation_methods = [
    Aggregation.MEAN,
    Aggregation.MEDIAN,
    Aggregation.MAXIMUM,
]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=agg_method,
        seed=42
    )
    detector.fit(x_train)
    p_vals = detector.predict(x_test, raw=False)

    # Apply FDR control
    fdr_controlled = false_discovery_control(p_vals, method='bh')
    print(f"{agg_method.value} aggregation: {(fdr_controlled < 0.05).sum()} detections")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Plot p-value distribution (visualization only - use FDR-controlled decisions for actual detection)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(p_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0.05, color='red', linestyle='--', label='α=0.05 (reference)')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title('P-value Distribution')
plt.legend()

plt.subplot(1, 2, 2)
# Color by FDR-controlled discoveries, not raw p-values
plt.scatter(range(len(p_values)), p_values, c=discoveries,
            cmap='coolwarm', alpha=0.6)
plt.axhline(y=0.05, color='red', linestyle='--', label='α=0.05 (reference)')
plt.xlabel('Sample Index')
plt.ylabel('p-value')
plt.title('P-values with FDR-controlled Discoveries')
plt.legend()

plt.tight_layout()
plt.show()
```

## Next Steps

- Try [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation
