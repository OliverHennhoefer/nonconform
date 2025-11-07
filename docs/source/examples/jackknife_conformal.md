# Jackknife Conformal Detection

This example demonstrates how to use jackknife (leave-one-out) conformal prediction for anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer
from scipy.stats import false_discovery_control
from nonconform.detection import ConformalDetector
from nonconform.strategy import Jackknife
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

# Create jackknife strategy (leave-one-out)
jackknife_strategy = Jackknife()

# Initialize detector with jackknife strategy
detector = ConformalDetector(
    detector=base_detector,
    strategy=jackknife_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42,
)

# Fit and predict
detector.fit(X)
p_values = detector.predict(X, raw=False)

# Detect anomalies
anomalies = p_values < 0.05
print(f"Number of anomalies detected: {anomalies.sum()}")
```

## Jackknife Plus Mode

```python
# Use plus mode to retain all models
jackknife_plus_strategy = Jackknife(plus=True)

detector_plus = ConformalDetector(
    detector=base_detector,
    strategy=jackknife_plus_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and predict with ensemble
detector_plus.fit(X)
p_values_plus = detector_plus.predict(X, raw=False)

print(f"Jackknife detections: {(p_values < 0.05).sum()}")
print(f"Jackknife+ detections: {(p_values_plus < 0.05).sum()}")
```

## FDR Control

```python
# Apply FDR control to jackknife p-values
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

print(f"\nFDR Control Results:")
print(f"Discoveries: {discoveries.sum()}")
print(f"Original detections: {(p_values < 0.05).sum()}")
print(f"False discovery reduction: {((p_values < 0.05).sum() - discoveries.sum())}")
```

## Computational Considerations

```python
import time

# Jackknife can be computationally expensive for large datasets
# Let's compare computation time with other strategies
from nonconform.strategy import Split
from nonconform.strategy import CrossValidation

# Use a subset for timing comparison
X_subset = X[:100]  # Use first 100 samples

strategies = {
    'Split': Split(n_calib=0.2),
    '5-fold CV': CrossValidation(k=5),
    'Jackknife': Jackknife()
}

timing_results = {}
for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )

    start_time = time.time()
    detector.fit(X_subset)
    _ = detector.predict(X_subset, raw=False)
    end_time = time.time()

    timing_results[name] = end_time - start_time
    print(f"{name}: {timing_results[name]:.2f} seconds")

print(f"\nJackknife is {timing_results['Jackknife'] / timing_results['Split']:.1f}x slower than Split")
```

## Stability Analysis

```python
import matplotlib.pyplot as plt

# Analyze jackknife stability for small datasets
dataset_sizes = [50, 100, 200, 300]
jackknife_results = []
split_results = []

for size in dataset_sizes:
    X_sample = X[:size]

    # Jackknife
    jk_detector = ConformalDetector(
        detector=base_detector,
        strategy=Jackknife(),
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    jk_detector.fit(X_sample)
    jk_p_values = jk_detector.predict(X_sample, raw=False)
    jackknife_results.append((jk_p_values < 0.05).sum() / size)

    # Split for comparison
    split_detector = ConformalDetector(
        detector=base_detector,
        strategy=Split(calib_size=0.2),
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    split_detector.fit(X_sample)
    split_p_values = split_detector.predict(X_sample, raw=False)
    split_results.append((split_p_values < 0.05).sum() / size)

plt.figure(figsize=(8, 5))
plt.plot(dataset_sizes, jackknife_results, 'o-', label='Jackknife', linewidth=2)
plt.plot(dataset_sizes, split_results, 's--', label='Split', linewidth=2)
plt.xlabel('Dataset Size')
plt.ylabel('Detection Rate')
plt.title('Detection Rate vs Dataset Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Leave-K-Out Variation

```python
# While standard jackknife is leave-one-out, we can simulate leave-k-out
# using cross-validation with appropriate number of folds

# Leave-2-out approximation (using n_splits = n_samples/2)
n_samples = 100
X_small = X[:n_samples]

leave_k_out_configs = [
    {'k': 1, 'n_splits': n_samples},  # Standard jackknife
    {'k': 2, 'n_splits': n_samples // 2},  # Leave-2-out approximation
    {'k': 5, 'n_splits': n_samples // 5},  # Leave-5-out approximation
]

for config in leave_k_out_configs:
    if config['k'] == 1:
        strategy = Jackknife()
    else:
        strategy = CrossValidation(k=config['n_splits'])

    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    detector.fit(X_small)
    p_vals = detector.predict(X_small, raw=False)

    print(f"Leave-{config['k']}-out: {(p_vals < 0.05).sum()} detections")
```

## Comparison with Other Strategies

```python
from nonconform.strategy import Bootstrap

# Comprehensive comparison
strategies = {
    'Jackknife': Jackknife(),
    'Jackknife+': Jackknife(plus=True),
    'Split': Split(calib_size=0.2),
    '5-fold CV': CrossValidation(k=5),
    'Bootstrap': Bootstrap(n_bootstraps=100, resampling_ratio=0.8)
}

comparison_results = {}
for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    detector.fit(X)
    p_vals = detector.predict(X, raw=False)

    # Apply FDR control
    adj_p_vals = false_discovery_control(p_vals, method='bh')

    comparison_results[name] = {
        'raw_detections': (p_vals < 0.05).sum(),
        'fdr_detections': (adj_p_vals < 0.05).sum(),
        'mean_p': p_vals.mean(),
        'std_p': p_vals.std()
    }

print("\nStrategy Comparison:")
print("-" * 80)
print(f"{'Strategy':<15} {'Raw Det.':<10} {'FDR Det.':<10} {'Mean p':<10} {'Std p':<10}")
print("-" * 80)
for name, results in comparison_results.items():
    print(f"{name:<15} {results['raw_detections']:<10} {results['fdr_detections']:<10} "
          f"{results['mean_p']:<10.3f} {results['std_p']:<10.3f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [cross-validation detection](cross_val_conformal.md) for robust calibration
