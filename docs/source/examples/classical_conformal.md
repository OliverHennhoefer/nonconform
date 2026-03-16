# Classical Conformal Anomaly Detection

This example demonstrates how to use classical conformal prediction for anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power
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
    aggregation="median",
    seed=42
)

# Fit the detector on training data (normal samples only)
detector.fit(x_train)

# Get raw anomaly scores (optional)
scores = detector.score_samples(x_test)

# Apply FDR-controlled selection
discoveries = detector.select(x_test, alpha=0.05)
p_values = detector.last_result.p_values

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
    aggregation="median",
    seed=42
)

# Fit and predict with cross-validation
cv_detector.fit(x_train)
cv_discoveries = cv_detector.select(x_test, alpha=0.05)

# Compare with split strategy
print(f"Split strategy detections: {discoveries.sum()}")
print(f"Cross-validation detections: {cv_discoveries.sum()}")
```

## Comparing Different Aggregation Methods

```python
# Try different aggregation methods
aggregation_methods = [
    "mean",
    "median",
    "maximum",
]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=agg_method,
        seed=42
    )
    detector.fit(x_train)
    selected = detector.select(x_test, alpha=0.05)
    print(f"{agg_method} aggregation: {selected.sum()} detections")
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
