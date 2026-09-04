---
description: "Compare split, cross-validation, leave-one-out, and bootstrap conformalization with measured fit cost and labeled discovery metrics."
---

# Data-efficient resampling

Resampling strategies construct calibration scores without reserving one fixed
holdout. They can improve data use when reference data is scarce, at the cost
of more model fits, retained models, and a different validity argument.

!!! important "The `+` name is not a transferable guarantee"

    In plus mode, `nonconform` retains resampling models and aggregates their
    raw test scores. Prediction-interval theorems for CV+, jackknife+, and JaB+
    do not automatically certify this anomaly-score aggregation. Keep `Split`
    as the clean finite-sample baseline and state the exact construction used.

## Complete comparison

```python
from time import perf_counter

import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import (
    ConformalDetector,
    CrossValidation,
    JackknifeBootstrap,
    Split,
)
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(80, 4))
x_test = np.vstack(
    [rng.normal(size=(16, 4)), rng.normal(loc=4.5, size=(4, 4))]
)
y_test = np.r_[np.zeros(16, dtype=int), np.ones(4, dtype=int)]

strategies = {
    "split": Split(n_calib=0.25),
    "cv_plus": CrossValidation(k=4, mode="plus"),
    "jackknife_plus": CrossValidation.jackknife(mode="plus"),
    "bootstrap_plus": JackknifeBootstrap(n_bootstraps=10, mode="plus"),
}

for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=IsolationForest(
            n_estimators=10,
            max_samples=0.8,
            random_state=42,
        ),
        strategy=strategy,
        seed=42,
    )

    started = perf_counter()
    detector.fit(x_reference)
    fit_seconds = perf_counter() - started

    selected = np.asarray(detector.select(x_test, alpha=0.1))
    print(
        name,
        {
            "retained_models": len(detector.detector_set),
            "calibration_scores": len(detector.calibration_set),
            "fit_seconds": round(fit_seconds, 3),
            "discoveries": int(selected.sum()),
            "realized_fdp": float(false_discovery_rate(y_test, selected)),
            "power": float(statistical_power(y_test, selected)),
        },
    )
```

This small synthetic dataset makes leave-one-out fitting affordable. Its timing
does not predict production latency. Measure the actual detector, feature
dimension, reference size, and test-batch size on deployment hardware.

The split strategy has only 20 calibration scores in this example, so its
classical p-value grid is much coarser than the 80-score grids produced by the
resampling strategies. That difference is part of the comparison, not proof
that one strategy is universally better.

## Interpreting the mechanics

| Strategy | Calibration construction | Test scoring in plus mode |
|---|---|---|
| `Split` | Held-out scores from one fixed model | One model |
| `CrossValidation(k=4)` | One out-of-fold score per reference row | Median raw score across four retained models by default |
| `CrossValidation.jackknife()` | One leave-one-out score per reference row | Median raw score across 80 retained models here |
| `JackknifeBootstrap(10)` | Out-of-bag score aggregated per reference row | Median raw score across ten retained models by default |

`JackknifeBootstrap(aggregation_method=...)` controls aggregation of out-of-bag
calibration scores. `ConformalDetector(aggregation=...)` separately controls
aggregation of retained models' test scores.

## Evaluation discipline

- Compare strategies on untouched labeled data or prespecified simulations.
- Repeat stochastic strategies across seeds and report variability.
- Include fit time, scoring time, retained-model count, and peak memory.
- Do not choose the strategy on the same family used for the final reported
  FDP and power.
- Do not describe resampling as a fix for distribution shift.

See [Conformalization strategies](../user_guide/conformalization_strategies.md)
for exact implementation details and primary references, then
[Choosing strategies](../user_guide/choosing_strategies.md) for the decision
process.
