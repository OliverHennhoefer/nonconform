# Data-Efficient Resampling Conformal Detection

Split conformal is the clearest baseline: fit on one subset, calibrate on
another, then test on new data. When the dataset is small, that holdout can be
expensive. Resampling strategies use the same data more efficiently by rotating
which observations are used for fitting and calibration.

This chapter covers the resampling family together:

- `CrossValidation(k=..., mode="plus")` for CV-style calibration
- `CrossValidation.jackknife(mode="plus")` for leave-one-out Jackknife+
- `JackknifeBootstrap(..., mode="plus")` for Jackknife+-after-Bootstrap (JaB+)

These methods often work well in practice, especially when training data is
scarce. Their guarantees are not interchangeable with the clean split-conformal
finite-sample guarantee; use `mode="plus"` when validity is more important than
inference-time memory.

## Setup

```python
import numpy as np
from pyod.models.lof import LOF
from sklearn.datasets import load_breast_cancer

from nonconform import ConformalDetector, CrossValidation, JackknifeBootstrap, Split
from nonconform.metrics import false_discovery_rate, statistical_power

data = load_breast_cancer()
X = data.data
y = data.target

# In this dataset, target=0 is malignant and target=1 is benign.
y_anomaly = 1 - y
base_detector = LOF()
```

## Choose a Resampling Strategy

| Strategy | Use When | Main Cost |
|----------|----------|-----------|
| CV+ | You want a practical small-data default | Fit one model per fold |
| Jackknife+ | The dataset is very small and every point matters | Fit one model per observation |
| JaB+ | You want bootstrap stability and can spend more compute | Fit many bootstrap models |

CV+ is usually the first resampling strategy to try. Jackknife+ pushes data use
further but is expensive. JaB+ is useful when bootstrap stability is valuable or
when you prefer a configurable ensemble size.

## CV+

```python
cv_strategy = CrossValidation(k=5, mode="plus")

cv_detector = ConformalDetector(
    detector=base_detector,
    strategy=cv_strategy,
    aggregation="median",
    seed=42,
)

cv_detector.fit(X)
cv_discoveries = cv_detector.select(X, alpha=0.05)

print(f"CV+ discoveries: {cv_discoveries.sum()}")
print(f"Empirical FDR: {false_discovery_rate(y_anomaly, cv_discoveries):.3f}")
print(f"Power: {statistical_power(y_anomaly, cv_discoveries):.3f}")
```

For smaller datasets, increase `k` if compute allows:

```python
for k in [3, 5, 10]:
    detector = ConformalDetector(
        detector=LOF(),
        strategy=CrossValidation(k=k, mode="plus"),
        aggregation="median",
        seed=42,
    )
    detector.fit(X)
    discoveries = detector.select(X, alpha=0.05)
    print(f"{k}-fold CV+: {discoveries.sum()} discoveries")
```

## Jackknife+

```python
jackknife_strategy = CrossValidation.jackknife(mode="plus")

jackknife_detector = ConformalDetector(
    detector=base_detector,
    strategy=jackknife_strategy,
    aggregation="median",
    seed=42,
)

jackknife_detector.fit(X)
jackknife_discoveries = jackknife_detector.select(X, alpha=0.05)

print(f"Jackknife+ discoveries: {jackknife_discoveries.sum()}")
```

Jackknife+ is leave-one-out CV+. It is the most data-intensive option in this
family and can be impractical once the training set is more than a few hundred
observations.

## JaB+

```python
jab_strategy = JackknifeBootstrap(n_bootstraps=100, mode="plus")

jab_detector = ConformalDetector(
    detector=base_detector,
    strategy=jab_strategy,
    aggregation="median",
    seed=42,
)

jab_detector.fit(X)
jab_discoveries = jab_detector.select(X, alpha=0.05)

print(f"JaB+ discoveries: {jab_discoveries.sum()}")
```

Use more bootstraps for smoother behavior when compute permits:

```python
for n_bootstraps in [50, 100, 200]:
    detector = ConformalDetector(
        detector=LOF(),
        strategy=JackknifeBootstrap(n_bootstraps=n_bootstraps, mode="plus"),
        aggregation="median",
        seed=42,
    )
    detector.fit(X)
    discoveries = detector.select(X, alpha=0.05)
    print(f"JaB+ with {n_bootstraps} bootstraps: {discoveries.sum()} discoveries")
```

## Compare Against Split

Always keep split conformal in the comparison. It is simpler, faster, and has
the cleanest finite-sample validity story.

```python
strategies = {
    "Split": Split(n_calib=0.2),
    "CV+": CrossValidation(k=5, mode="plus"),
    "Jackknife+": CrossValidation.jackknife(mode="plus"),
    "JaB+": JackknifeBootstrap(n_bootstraps=100, mode="plus"),
}

results = {}
for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=LOF(),
        strategy=strategy,
        aggregation="median",
        seed=42,
    )
    detector.fit(X)
    p_values = detector.compute_p_values(X)
    discoveries = detector.select(X, alpha=0.05)
    results[name] = {
        "discoveries": discoveries.sum(),
        "mean_p": p_values.mean(),
        "empirical_fdr": false_discovery_rate(y_anomaly, discoveries),
        "power": statistical_power(y_anomaly, discoveries),
    }

print("\nStrategy comparison:")
print(f"{'Strategy':<12} {'Discoveries':<12} {'Mean p':<10} {'Emp. FDR':<10} {'Power':<10}")
for name, values in results.items():
    print(
        f"{name:<12} {values['discoveries']:<12} "
        f"{values['mean_p']:<10.3f} {values['empirical_fdr']:<10.3f} "
        f"{values['power']:<10.3f}"
    )
```

## Practical Guidance

- Start with `Split` if you have enough data for a clean train/calibration split.
- Try CV+ when the holdout split costs too much power.
- Try Jackknife+ only when the dataset is small enough for leave-one-out fitting.
- Try JaB+ when bootstrap stability matters and the extra compute is acceptable.
- Prefer `mode="plus"` for resampling strategies unless inference-time memory is
  the main constraint.

## Next Steps

- Use [classical conformal detection](classical_conformal.md) as the split baseline.
- Read [conformalization strategies](../user_guide/conformalization_strategies.md) for the guarantee and mode details.
- Learn [FDR control](fdr_control.md) before interpreting many simultaneous discoveries.
