# Jackknife+-after-Bootstrap (JaB+) Conformal Detection

This example demonstrates how to use bootstrap resampling for conformal anomaly detection using the JaB+ strategy.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer
from scipy.stats import false_discovery_control
from nonconform import Aggregation, ConformalDetector, JackknifeBootstrap, false_discovery_rate, statistical_power

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
    seed=42
)

# Fit and predict
detector.fit(X)
p_values = detector.predict(X, raw=False)

# Apply FDR control (Benjamini-Hochberg)
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05
print(f"Discoveries with FDR control: {discoveries.sum()}")
```

## Plus Mode for JaB+

```python
# Use plus mode to keep all bootstrap models for aggregation
jab_plus_strategy = JackknifeBootstrap(
    n_bootstraps=100,
    aggregation_method=Aggregation.MEDIAN,
    plus=True
)

detector_plus = ConformalDetector(
    detector=base_detector,
    strategy=jab_plus_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and predict with ensemble
detector_plus.fit(X)
p_values_plus = detector_plus.predict(X, raw=False)

# Compare with FDR control
jab_disc = false_discovery_control(p_values, method='bh') < 0.05
jab_plus_disc = false_discovery_control(p_values_plus, method='bh') < 0.05
print(f"JaB+ discoveries: {jab_disc.sum()}")
print(f"JaB+ (plus) discoveries: {jab_plus_disc.sum()}")
```

## Comparing Different Bootstrap Configurations

```python
# Try different bootstrap configurations
bootstrap_counts = [50, 100, 200]

results = {}
for n_bootstraps in bootstrap_counts:
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

    key = f"B={n_bootstraps}"
    results[key] = disc.sum()
    print(f"{key}: {results[key]} discoveries")
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
        strategy=JackknifeBootstrap(n_bootstraps=50),
        aggregation=Aggregation.MEDIAN,
        seed=np.random.randint(1000)
    )
    det.fit(X)
    p_vals = det.predict(X, raw=False)
    disc = false_discovery_control(p_vals, method='bh') < 0.05
    stability_results.append(disc.sum())

plt.boxplot(stability_results)
plt.ylabel('Number of Detections')
plt.title('Bootstrap Detection Stability')

plt.tight_layout()
plt.show()
```

## Comparison with Other Strategies

```python
from nonconform import CrossValidation, JackknifeBootstrap, Split

# Compare strategies
strategies = {
    'JaB+': JackknifeBootstrap(n_bootstraps=50),
    'Split': Split(n_calib=0.2),
    'CV': CrossValidation(k=5)
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
    disc = false_discovery_control(p_vals, method='bh') < 0.05
    comparison_results[name] = {
        'discoveries': disc.sum(),
        'min_p': p_vals.min(),
        'mean_p': p_vals.mean()
    }

print("\nStrategy Comparison (with FDR control):")
for name, results in comparison_results.items():
    print(f"{name}:")
    print(f"  Discoveries: {results['discoveries']}")
    print(f"  Min p-value: {results['min_p']:.4f}")
    print(f"  Mean p-value: {results['mean_p']:.4f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [cross-validation detection](cross_val_conformal.md) for robust calibration
