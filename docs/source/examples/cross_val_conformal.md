# Cross-validation Conformal Detection

Use k-fold cross-validation for conformal anomaly detection.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer
from scipy.stats import false_discovery_control
from nonconform import Aggregation, ConformalDetector, CrossValidation, false_discovery_rate, statistical_power

# Load example data
data = load_breast_cancer()
X = data.data
y = data.target
```

## Basic Usage

```python
# Initialize base detector
base_detector = LOF()

# Create cross-validation strategy
cv_strategy = CrossValidation(k=5)

# Initialize detector with cross-validation strategy
detector = ConformalDetector(
    detector=base_detector,
    strategy=cv_strategy,
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

## Cross-Validation Plus Mode

```python
# Use plus mode to retain all fold models
cv_plus_strategy = CrossValidation(k=5, plus=True)

detector_plus = ConformalDetector(
    detector=base_detector,
    strategy=cv_plus_strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit and predict with ensemble
detector_plus.fit(X)
p_values_plus = detector_plus.predict(X, raw=False)

# Compare with FDR control
cv_disc = false_discovery_control(p_values, method='bh') < 0.05
cv_plus_disc = false_discovery_control(p_values_plus, method='bh') < 0.05
print(f"CV discoveries: {cv_disc.sum()}")
print(f"CV+ discoveries: {cv_plus_disc.sum()}")
```

## Comparing Different Numbers of Folds

```python
# Try different numbers of folds
fold_options = [3, 5, 10]

results = {}
for n_folds in fold_options:
    strategy = CrossValidation(k=n_folds)
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
    )
    detector.fit(X)
    p_vals = detector.predict(X, raw=False)
    disc = false_discovery_control(p_vals, method='bh') < 0.05

    results[f"{n_folds}-fold"] = disc.sum()
    print(f"{n_folds}-fold CV: {results[f'{n_folds}-fold']} discoveries")
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

## Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

# Create stratified CV strategy (useful when you have class labels)
# Note: In anomaly detection, we typically don't have labels during training,
# but this example shows how to use it if you do have some labeled data
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For demonstration, create synthetic labels based on anomaly scores
# In practice, you might have some labeled normal/anomaly data
temp_detector = LOF(contamination=0.1)
temp_detector.fit(X)
synthetic_labels = (temp_detector.decision_function(X) > np.percentile(temp_detector.decision_function(X), 90)).astype(int)

# Use stratified splits
for fold, (train_idx, val_idx) in enumerate(stratified_cv.split(X, synthetic_labels)):
    print(f"Fold {fold + 1}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
```

## Cross-Validation Stability Analysis

```python
import matplotlib.pyplot as plt

# Analyze stability across different random seeds
seeds = range(10)
cv_results = []

for seed in seeds:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=CrossValidation(k=5),
        aggregation=Aggregation.MEDIAN,
        seed=seed,
    )
    detector.fit(X)
    p_vals = detector.predict(X, raw=False)
    disc = false_discovery_control(p_vals, method='bh') < 0.05
    cv_results.append(disc.sum())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(seeds, cv_results, 'o-')
plt.xlabel('Random Seed')
plt.ylabel('Number of Detections')
plt.title('CV Detection Stability Across Seeds')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(cv_results)
plt.ylabel('Number of Detections')
plt.title('CV Detection Distribution')
plt.xticks([1], ['5-fold CV'])

plt.tight_layout()
plt.show()

print(f"Mean detections: {np.mean(cv_results):.1f}")
print(f"Std detections: {np.std(cv_results):.1f}")
```

## Comparison with Other Strategies

```python
from nonconform import Split, JackknifeBootstrap

# Compare cross-validation with other strategies
strategies = {
    'Split': Split(n_calib=0.2),
    '5-fold CV': CrossValidation(k=5),
    '10-fold CV': CrossValidation(k=10),
    'JaB+': JackknifeBootstrap(n_bootstraps=50),
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
        'min_p': p_vals.min(),
        'mean_p': p_vals.mean()
    }

print("\nStrategy Comparison (with FDR control):")
print("-" * 55)
print(f"{'Strategy':<15} {'Discoveries':<12} {'Min p-val':<12} {'Mean p-val':<12}")
print("-" * 55)
for name, results in comparison_results.items():
    print(f"{name:<15} {results['discoveries']:<12} "
          f"{results['min_p']:<12.4f} {results['mean_p']:<12.4f}")
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation
