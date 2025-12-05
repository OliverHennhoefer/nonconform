# Jackknife+-after-Bootstrap (JaB+) Conformal Detection

This example demonstrates how to use the JaB+ strategy for conformal anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer
from scipy.stats import false_discovery_control
from nonconform import Aggregation, ConformalDetector, JackknifeBootstrap, Split, CrossValidation, false_discovery_rate, statistical_power

# Load example data
data = load_breast_cancer()
X = data.data
y = data.target
```

## Basic Usage

```python
# Initialize base detector
base_detector = LOF()

# Create JaB+ strategy
jab_strategy = JackknifeBootstrap(n_bootstraps=50)

# Initialize detector with JaB+ strategy
detector = ConformalDetector(
    detector=base_detector,
    strategy=jab_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42,
)

# Fit and predict
detector.fit(X)
p_values = detector.predict(X, raw=False)

# Apply FDR control (Benjamini-Hochberg)
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05
print(f"Discoveries with FDR control: {discoveries.sum()}")
```

## Evaluation Metrics

```python
# With ground truth labels available (y from breast cancer dataset)
# Note: In breast cancer, target=0 is malignant (anomaly), target=1 is benign (normal)
y_anomaly = 1 - y  # Convert so 1 = anomaly

print(f"\nEvaluation with FDR Control:")
print(f"Discoveries: {discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y=y_anomaly, y_hat=discoveries):.3f}")
print(f"Statistical Power: {statistical_power(y=y_anomaly, y_hat=discoveries):.3f}")
```

## Comparing Different Bootstrap Configurations

```python
# Try different numbers of bootstraps
bootstrap_options = [25, 50, 100]

results = {}
for n_bootstraps in bootstrap_options:
    strategy = JackknifeBootstrap(n_bootstraps=n_bootstraps)
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    detector.fit(X)
    p_vals = detector.predict(X, raw=False)
    disc = false_discovery_control(p_vals, method='bh') < 0.05

    results[f"{n_bootstraps} bootstraps"] = disc.sum()
    print(f"{n_bootstraps} bootstraps: {results[f'{n_bootstraps} bootstraps']} discoveries")
```

## Computational Considerations

```python
import time

# JaB+ computation time comparison with other strategies
X_subset = X[:100]  # Use first 100 samples

strategies = {
    'Split': Split(n_calib=0.2),
    '5-fold CV': CrossValidation(k=5),
    'JaB+ (50)': JackknifeBootstrap(n_bootstraps=50)
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
```

## Stability Analysis

```python
import matplotlib.pyplot as plt

# Analyze JaB+ stability for small datasets
dataset_sizes = [50, 100, 200, 300]
jab_results = []
split_results = []

for size in dataset_sizes:
    X_sample = X[:size]

    # JaB+
    jab_detector = ConformalDetector(
        detector=base_detector,
        strategy=JackknifeBootstrap(n_bootstraps=50),
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    jab_detector.fit(X_sample)
    jab_p_values = jab_detector.predict(X_sample, raw=False)
    jab_disc = false_discovery_control(jab_p_values, method='bh') < 0.05
    jab_results.append(jab_disc.sum() / size)

    # Split for comparison
    split_detector = ConformalDetector(
        detector=base_detector,
        strategy=Split(n_calib=0.2),
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    split_detector.fit(X_sample)
    split_p_values = split_detector.predict(X_sample, raw=False)
    split_disc = false_discovery_control(split_p_values, method='bh') < 0.05
    split_results.append(split_disc.sum() / size)

plt.figure(figsize=(8, 5))
plt.plot(dataset_sizes, jab_results, 'o-', label='JaB+', linewidth=2)
plt.plot(dataset_sizes, split_results, 's--', label='Split', linewidth=2)
plt.xlabel('Dataset Size')
plt.ylabel('Discovery Rate')
plt.title('Discovery Rate vs Dataset Size (with FDR control)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Comparison with Other Strategies

```python
# Comprehensive comparison
strategies = {
    'Split': Split(n_calib=0.2),
    '5-fold CV': CrossValidation(k=5),
    'JaB+ (25)': JackknifeBootstrap(n_bootstraps=25),
    'JaB+ (50)': JackknifeBootstrap(n_bootstraps=50),
    'JaB+ (100)': JackknifeBootstrap(n_bootstraps=100),
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
    disc = false_discovery_control(p_vals, method='bh') < 0.05

    comparison_results[name] = {
        'discoveries': disc.sum(),
        'mean_p': p_vals.mean(),
        'std_p': p_vals.std()
    }

print("\nStrategy Comparison (with FDR control):")
print("-" * 55)
print(f"{'Strategy':<15} {'Discoveries':<12} {'Mean p':<12} {'Std p':<12}")
print("-" * 55)
for name, results in comparison_results.items():
    print(f"{name:<15} {results['discoveries']:<12} "
          f"{results['mean_p']:<12.3f} {results['std_p']:<12.3f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [cross-validation detection](cross_val_conformal.md) for robust calibration
