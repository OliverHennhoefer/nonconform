# Conditional Conformal Selection

This example shows how to use conditionally calibrated conformal p-values with
FDR-controlled selection via `detector.select(...)`.

## Setup

```python
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power
from nonconform.scoring import ConditionalEmpirical

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)
```

## Conditional Calibration + Selection

```python
detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=1_000),
    estimation=ConditionalEmpirical(
        method="simes",
        delta=0.1,
        tie_break="classical",
    ),
    seed=1,
)

detector.fit(x_train)
discoveries = detector.select(x_test, alpha=0.2)

# Access p-values from the same selection pass
p_values = detector.last_result.p_values
print(f"P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=discoveries):.3f}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=discoveries):.3f}")
```

## Method Variants

```python
methods = ["mc", "simes", "dkwm", "asymptotic"]

for method in methods:
    detector = ConformalDetector(
        detector=IForest(),
        strategy=Split(n_calib=1_000),
        estimation=ConditionalEmpirical(
            method=method,
            delta=0.1,
            tie_break="classical",
        ),
        seed=1,
    )
    detector.fit(x_train)
    discoveries = detector.select(x_test, alpha=0.2)
    p_values = detector.last_result.p_values
    print(f"{method}: {discoveries.sum()} discoveries, min p={p_values.min():.4f}")
```
