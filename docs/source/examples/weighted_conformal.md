---
description: "Compare standard BH selection with weighted conformal p-values and WCS under a controlled covariate-shift simulation."
---

# Weighted conformal selection

This example simulates a shifted target null, compares standard split-conformal
BH selection with weighted conformalized selection (WCS), and inspects the
estimated importance weights.

It is an API and evaluation example, not evidence that estimated weighting is
valid for every real shift.

## Complete comparison

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.fdr import Pruning
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(3_000, 4))
x_target = np.vstack(
    [
        rng.normal(loc=0.5, size=(24, 4)),
        rng.normal(loc=6.0, size=(6, 4)),
    ]
)
y_target = np.r_[np.zeros(24, dtype=int), np.ones(6, dtype=int)]

standard = ConformalDetector(
    detector=IsolationForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.4),
    seed=42,
).fit(x_reference)
standard_selected = np.asarray(standard.select(x_target, alpha=0.1))
standard_result = standard.last_result
assert standard_result is not None
assert standard_result.p_values is not None

weighted = ConformalDetector(
    detector=IsolationForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.4),
    weight_estimator=logistic_weight_estimator(clip_quantile=0.05),
    seed=42,
).fit(x_reference)
weighted_selected = np.asarray(
    weighted.select(
        x_target,
        alpha=0.1,
        pruning=Pruning.DETERMINISTIC,
    )
)
weighted_result = weighted.last_result
assert weighted_result is not None
assert weighted_result.p_values is not None
assert weighted_result.calib_weights is not None
assert weighted_result.test_weights is not None

def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())

for name, selected in {
    "standard_bh": standard_selected,
    "weighted_wcs": weighted_selected,
}.items():
    print(
        name,
        {
            "discoveries": int(selected.sum()),
            "realized_fdp": float(false_discovery_rate(y_target, selected)),
            "power": float(statistical_power(y_target, selected)),
        },
    )

print(
    "calibration weights:",
    {
        "quantiles": np.quantile(
            weighted_result.calib_weights,
            [0.0, 0.05, 0.5, 0.95, 1.0],
        ),
        "ess": effective_sample_size(weighted_result.calib_weights),
    },
)
print(
    "target weights:",
    {
        "quantiles": np.quantile(
            weighted_result.test_weights,
            [0.0, 0.05, 0.5, 0.95, 1.0],
        ),
        "ess": effective_sample_size(weighted_result.test_weights),
    },
)
```

The standard and weighted calls intentionally use different selection
procedures. Weighted conformal p-values need not satisfy the positive
dependence used by ordinary BH, so weighted `select(...)` dispatches to WCS.

## What must be true outside this simulation

Weighted validity requires a defensible covariate-shift model, target support
inside calibration support, a fixed score construction, and suitable density
ratios. Here the shift and labels are generated explicitly. In an application,
those properties require domain justification and held-out diagnostics.

`clip_quantile=0.05` truncates the combined estimated weights at their 5th and
95th percentiles. This can stabilize estimation but changes the ratio. Compare
clipping choices on separate validation scenarios rather than selecting the
most attractive final result.

ESS and quantiles are descriptive diagnostics, not universal validity tests.
Near-separation of calibration and target samples, highly concentrated weights,
or strong sensitivity to the estimator signals weak effective overlap.

See [Weighted conformal inference](../user_guide/weighted_conformal.md) for the
formulas, batch-state rules, bagged estimator, pruning modes, and references.
