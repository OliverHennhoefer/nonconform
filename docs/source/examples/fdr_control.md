# FDR Control for Multiple Testing

This example demonstrates how to use False Discovery Rate (FDR) control in anomaly detection using scipy.stats.false_discovery_control.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer, make_blobs
from scipy.stats import false_discovery_control
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation
from nonconform.utils.stat.metrics import false_discovery_rate

# Load example data
data = load_breast_cancer()
X = data.data
y = data.target
```

## Basic Usage

```python
# Initialize detector
base_detector = LOF()
strategy = Split(n_calib=0.2)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and get p-values
detector.fit(X)
p_values = detector.predict(X, raw=False)

# Apply FDR control using scipy
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

print(f"Original detections: {(p_values < 0.05).sum()}")
print(f"FDR-controlled discoveries: {discoveries.sum()}")
print(f"Reduction: {(p_values < 0.05).sum() - discoveries.sum()}")
```

## Different FDR Control Methods

```python
# Available methods in scipy.stats.false_discovery_control
fdr_methods = ['bh', 'by']

results = {}
for method in fdr_methods:
    adjusted_p_vals = false_discovery_control(p_values, method=method)
    discoveries = adjusted_p_vals < 0.05
    results[method] = discoveries.sum()

    print(f"{method.upper()} method: {results[method]} discoveries")

# Compare with no adjustment
no_adjustment = (p_values < 0.05).sum()
print(f"No adjustment: {no_adjustment} detections")
```

## Weighted Conformal Selection (Covariate Shift)

```python
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import JackknifeBootstrap
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import weighted_false_discovery_control
from pyod.models.iforest import IForest

# Load a dataset that exhibits covariate shift between calibration and test sets
x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

weighted_detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=JackknifeBootstrap(n_bootstraps=50),
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=1),
    seed=1,
)

weighted_detector.fit(x_train)

# Obtain weighted p-values, raw scores, and importance weights
weighted_p_values = weighted_detector.predict(x_test, raw=False)

# Weighted Conformal Selection controls the FDR under covariate shift
wcs_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=1,
)
# detector.last_result bundles the scores/weights produced by predict()

print(f"WCS detections: {wcs_mask.sum()} out of {len(wcs_mask)} test points")
print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=wcs_mask):.3f}")
```

## FDR Control at Different Levels

```python
# Try different FDR levels
fdr_levels = [0.01, 0.05, 0.1, 0.2]

print("\nFDR Control at Different Levels:")
print("-" * 40)
print(f"{'FDR Level':<12} {'Discoveries':<12} {'Adjusted α':<12}")
print("-" * 40)

for alpha in fdr_levels:
    adjusted_p_vals = false_discovery_control(p_values, method="bh", alpha=alpha)
    discoveries = adjusted_p_vals < alpha

    print(f"{alpha:<12} {discoveries.sum():<12} {adjusted_p_vals.min():.6f}")
```

## Evaluating FDR Control Performance

```python
# Create synthetic data with known ground truth
np.random.seed(42)

# Generate normal data
X_normal, _ = make_blobs(n_samples=800, centers=1, cluster_std=1.0, random_state=42)

# Generate anomalies
X_anomalies = np.random.uniform(-6, 6, (200, X_normal.shape[1]))

# Combine data
X_combined = np.vstack([X_normal, X_anomalies])
y_true = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomalies))])

# Fit detector and get p-values
detector.fit(X_normal)  # Fit only on normal data
p_values = detector.predict(X_combined, raw=False)

# Apply different FDR control levels
fdr_levels = [0.05, 0.1, 0.15, 0.2]

print("\nFDR Control Performance Evaluation:")
print("-" * 80)
print(f"{'FDR Level':<10} {'Discoveries':<12} {'True Pos':<10} {'False Pos':<10} {'Precision':<10} {'Empirical FDR':<12}")
print("-" * 80)

for alpha in fdr_levels:
    adjusted_p_vals = false_discovery_control(p_values, method="bh", alpha=alpha)
    discoveries = adjusted_p_vals < alpha

    true_positives = np.sum(discoveries & (y_true == 1))
    false_positives = np.sum(discoveries & (y_true == 0))
    precision = true_positives / max(1, discoveries.sum())
    empirical_fdr = false_positives / max(1, discoveries.sum())

    print(f"{alpha:<10} {discoveries.sum():<12} {true_positives:<10} {false_positives:<10} "
          f"{precision:<10.3f} {empirical_fdr:<12.3f}")
```

## Multiple Detectors with FDR Control

```python
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM

# Multiple detectors
detectors = {
    'LOF': LOF(contamination=0.1),
    'KNN': KNN(contamination=0.1),
    'OCSVM': OCSVM(contamination=0.1)
}

# Get p-values from each detector
all_p_values = {}
strategy = Split(n_calib=0.2)

for name, base_det in detectors.items():
    detector = ConformalDetector(
        detector=base_det,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    detector.fit(X_normal)
    p_vals = detector.predict(X_combined, raw=False)
    all_p_values[name] = p_vals

# Apply FDR control to each detector's p-values
print("\nMultiple Detectors with FDR Control:")
print("-" * 60)
print(f"{'Detector':<10} {'Raw Det.':<10} {'FDR Det.':<10} {'True Pos':<10} {'Precision':<10}")
print("-" * 60)

for name, p_vals in all_p_values.items():
    # Raw detections
    raw_detections = (p_vals < 0.05).sum()

    # FDR controlled detections
    adj_p_vals = false_discovery_control(p_vals, method='bh', alpha=0.05)
    fdr_discoveries = adj_p_vals < 0.05

    # Performance metrics
    true_pos = np.sum(fdr_discoveries & (y_true == 1))
    precision = true_pos / max(1, fdr_discoveries.sum())

    print(f"{name:<10} {raw_detections:<10} {fdr_discoveries.sum():<10} {true_pos:<10} {precision:<10.3f}")
```

## Ensemble with FDR Control

```python
# Combine p-values from multiple detectors and apply FDR control
# Using Fisher's method for combining p-values
from scipy.stats import combine_pvalues

# Combine p-values using Fisher's method
p_values_list = list(all_p_values.values())
combined_stats, combined_p_values = combine_pvalues(np.array(p_values_list).T, method='fisher')

# Apply FDR control to combined p-values
adj_combined_p_vals = false_discovery_control(combined_p_values, method='bh', alpha=0.05)
combined_discoveries = adj_combined_p_vals < 0.05

# Evaluate ensemble performance
ensemble_true_pos = np.sum(combined_discoveries & (y_true == 1))
ensemble_false_pos = np.sum(combined_discoveries & (y_true == 0))
ensemble_precision = ensemble_true_pos / max(1, combined_discoveries.sum())

print(f"\nEnsemble with FDR Control:")
print(f"Discoveries: {combined_discoveries.sum()}")
print(f"True Positives: {ensemble_true_pos}")
print(f"False Positives: {ensemble_false_pos}")
print(f"Precision: {ensemble_precision:.3f}")
print(f"Empirical FDR: {ensemble_false_pos / max(1, combined_discoveries.sum()):.3f}")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Visualize FDR control effects
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# P-value histogram
axes[0, 0].hist(p_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
axes[0, 0].set_xlabel('p-value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('P-value Distribution')
axes[0, 0].legend()

# Adjusted p-value histogram
adjusted_p_vals = false_discovery_control(p_values, method='bh', alpha=0.05)
axes[0, 1].hist(adjusted_p_vals, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[0, 1].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
axes[0, 1].set_xlabel('Adjusted p-value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('BH Adjusted P-value Distribution')
axes[0, 1].legend()

# Comparison of detection methods
detection_methods = ['Raw (α=0.05)', 'BH FDR Control', 'BY FDR Control']
detection_counts = [
    (p_values < 0.05).sum(),
    (false_discovery_control(p_values, method='bh') < 0.05).sum(),
    (false_discovery_control(p_values, method='by') < 0.05).sum()
]

axes[1, 0].bar(detection_methods, detection_counts, color=['blue', 'orange', 'green'])
axes[1, 0].set_ylabel('Number of Detections')
axes[1, 0].set_title('Detection Comparison')
axes[1, 0].tick_params(axis='x', rotation=45)

# FDR control at different levels
fdr_levels = np.arange(0.01, 0.21, 0.01)
discoveries_at_levels = []

for alpha in fdr_levels:
    adj_p_vals = false_discovery_control(p_values, method='bh', alpha=alpha)
    discoveries_at_levels.append((adj_p_vals < alpha).sum())

axes[1, 1].plot(fdr_levels, discoveries_at_levels, 'o-', linewidth=2)
axes[1, 1].axhline(y=(p_values < 0.05).sum(), color='red', linestyle='--',
                   label='Raw (α=0.05)')
axes[1, 1].set_xlabel('FDR Level')
axes[1, 1].set_ylabel('Number of Discoveries')
axes[1, 1].set_title('Discoveries vs FDR Level')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Power Analysis

```python
# Analyze statistical power under different scenarios
effect_sizes = [0.5, 1.0, 1.5, 2.0]  # Distance between normal and anomaly centers
power_results = {}

for effect_size in effect_sizes:
    # Generate data with specified effect size
    X_norm = np.random.normal(0, 1, (500, 2))
    X_anom = np.random.normal(effect_size, 1, (100, 2))
    X_test = np.vstack([X_norm, X_anom])
    y_test = np.hstack([np.zeros(500), np.ones(100)])

    # Fit detector and get p-values
    detector = ConformalDetector(
        detector=LOF(contamination=0.1),
        strategy=SplitStrategy(calibration_size=0.2),
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    detector.fit(X_norm)
    p_vals = detector.predict(X_test, raw=False)

    # Apply FDR control
    adj_p_vals = false_discovery_control(p_vals, method='bh', alpha=0.05)
    discoveries = adj_p_vals < 0.05

    # Calculate power (true positive rate)
    power = np.sum(discoveries & (y_test == 1)) / np.sum(y_test == 1)
    power_results[effect_size] = power

print("\nPower Analysis:")
print("-" * 30)
print(f"{'Effect Size':<12} {'Power':<8}")
print("-" * 30)
for effect_size, power in power_results.items():
    print(f"{effect_size:<12} {power:<8.3f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation
