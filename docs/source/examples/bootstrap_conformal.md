# Bootstrap-based Conformal Detection

This example demonstrates how to use bootstrap resampling for conformal anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer
from scipy.stats import false_discovery_control
from nonconform.detection import ConformalDetector
from nonconform.strategy import Bootstrap
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

# Create bootstrap strategy
bootstrap_strategy = Bootstrap(
    n_bootstraps=100,
    resampling_ratio=0.8
)

# Initialize detector with bootstrap strategy
detector = ConformalDetector(
    detector=base_detector,
    strategy=bootstrap_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and predict
detector.fit(X)
p_values = detector.predict(X, raw=False)

# Detect anomalies
anomalies = p_values < 0.05
print(f"Number of anomalies detected: {anomalies.sum()}")
```

## Bootstrap Plus Mode

```python
# Use bootstrap plus mode for better calibration
bootstrap_plus_strategy = Bootstrap(
    n_bootstraps=100,
    resampling_ratio=0.8,
    plus=True
)

detector_plus = ConformalDetector(
    detector=base_detector,
    strategy=bootstrap_plus_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and predict with ensemble
detector_plus.fit(X)
p_values_plus = detector_plus.predict(X, raw=False)

print(f"Bootstrap detections: {(p_values < 0.05).sum()}")
print(f"Bootstrap+ detections: {(p_values_plus < 0.05).sum()}")
```

## Comparing Different Bootstrap Configurations

```python
# Try different bootstrap configurations
configurations = [
    {"n_bootstraps": 50, "resampling_ratio": 0.7},
    {"n_bootstraps": 100, "resampling_ratio": 0.8},
    {"n_bootstraps": 200, "resampling_ratio": 0.9}
]

results = {}
for config in configurations:
    strategy = Bootstrap(**config)
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    detector.fit(X)
    p_vals = detector.predict(X, raw=False)

    key = f"B={config['n_bootstraps']}, r={config['resampling_ratio']}"
    results[key] = (p_vals < 0.05).sum()
    print(f"{key}: {results[key]} detections")
```

## FDR Control with Bootstrap

```python
# Apply FDR control to bootstrap p-values
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05

print(f"\nFDR Control Results:")
print(f"Discoveries: {discoveries.sum()}")
print(f"Original detections: {(p_values < 0.05).sum()}")
print(f"Reduction: {(p_values < 0.05).sum() - discoveries.sum()}")
```

## Accessing Bootstrap Iterations

```python
# Track calibration scores during bootstrap iterations
iteration_scores = []
iteration_stats = []

def track_bootstrap_iterations(iteration: int, scores: np.ndarray):
    """Callback to track calibration scores per iteration."""
    iteration_scores.append(scores.copy())
    iteration_stats.append({
        'iteration': iteration,
        'count': len(scores),
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max()
    })
    print(f"Iteration {iteration}: {len(scores)} calibration scores, "
          f"mean={scores.mean():.3f}")

# Initialize bootstrap strategy with callback
bootstrap_strategy = Bootstrap(
    n_bootstraps=10,
    resampling_ratio=0.8
)

detector = ConformalDetector(
    detector=base_detector,
    strategy=bootstrap_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit with iteration callback
detector.detector_set, detector.calibration_set = bootstrap_strategy.fit_calibrate(
    X, detector.detector, iteration_callback=track_bootstrap_iterations
)

# Analyze iteration progression
print(f"\nBootstrap completed with {len(iteration_scores)} iterations")
print(f"Total calibration scores: {len(detector.calibration_set)}")
```

## Bootstrap Iteration Analysis

```python
# Analyze how calibration distribution evolves
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# Plot calibration score evolution
plt.subplot(1, 3, 1)
for i, scores in enumerate(iteration_scores[:5]):  # Show first 5 iterations
    plt.hist(scores, bins=20, alpha=0.5, label=f'Iter {i}')
plt.xlabel('Calibration Score')
plt.ylabel('Frequency')
plt.title('Calibration Score Distribution per Iteration')
plt.legend()

# Plot iteration statistics
plt.subplot(1, 3, 2)
means = [stat['mean'] for stat in iteration_stats]
stds = [stat['std'] for stat in iteration_stats]
iterations = list(range(len(iteration_stats)))

plt.errorbar(iterations, means, yerr=stds, marker='o', capsize=5)
plt.xlabel('Bootstrap Iteration')
plt.ylabel('Mean Calibration Score')
plt.title('Score Statistics Evolution')

# Cumulative distribution analysis
plt.subplot(1, 3, 3)
cumulative_scores = []
for i in range(len(iteration_scores)):
    # Combine all scores up to iteration i
    combined = np.concatenate(iteration_scores[:i+1])
    cumulative_scores.append(combined.mean())

plt.plot(iterations, cumulative_scores, marker='s')
plt.xlabel('Bootstrap Iteration')
plt.ylabel('Cumulative Mean Score')
plt.title('Cumulative Calibration Mean')

plt.tight_layout()
plt.show()

# Print iteration summary
print("\nIteration Summary:")
for stat in iteration_stats:
    print(f"Iteration {stat['iteration']:2d}: "
          f"{stat['count']:3d} scores, "
          f"mean={stat['mean']:6.3f}, "
          f"std={stat['std']:6.3f}")
```

## Uncertainty Quantification

```python
# Get raw scores for uncertainty analysis
raw_scores = detector.predict(X, raw=True)

# Analyze score distribution
plt.figure(figsize=(12, 4))

# Score distribution
plt.subplot(1, 3, 1)
plt.hist(raw_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Bootstrap Anomaly Score Distribution')

# P-value calculation using final calibration set
p_values = detector.predict(X, raw=False)

# P-value vs Score relationship
plt.subplot(1, 3, 2)
plt.scatter(raw_scores, p_values, alpha=0.5)
plt.xlabel('Anomaly Score')
plt.ylabel('p-value')
plt.title('Score vs P-value Relationship')

# Bootstrap stability analysis
plt.subplot(1, 3, 3)
# Run multiple bootstrap iterations
stability_results = []
for _ in range(10):
    det = ConformalDetector(
        detector=base_detector,
        strategy=Bootstrap(n_bootstraps=50, resampling_ratio=0.8),
        aggregation=Aggregation.MEDIAN,
        seed=np.random.randint(1000)
    )
    det.fit(X)
    p_vals = det.predict(X, raw=False)
    stability_results.append((p_vals < 0.05).sum())

plt.boxplot(stability_results)
plt.ylabel('Number of Detections')
plt.title('Bootstrap Detection Stability')

plt.tight_layout()
plt.show()
```

## Comparison with Other Strategies

```python
from nonconform.strategy import Split, Jackknife

# Compare bootstrap with other strategies
strategies = {
    'Bootstrap': Bootstrap(n_bootstraps=100, resampling_ratio=0.8),
    'Split': Split(n_calib=0.2),
    'Jackknife': Jackknife()
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
    comparison_results[name] = {
        'detections': (p_vals < 0.05).sum(),
        'min_p': p_vals.min(),
        'mean_p': p_vals.mean()
    }

print("\nStrategy Comparison:")
for name, results in comparison_results.items():
    print(f"{name}:")
    print(f"  Detections: {results['detections']}")
    print(f"  Min p-value: {results['min_p']:.4f}")
    print(f"  Mean p-value: {results['mean_p']:.4f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [cross-validation detection](cross_val_conformal.md) for robust calibration
