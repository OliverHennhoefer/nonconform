---
description: "Compare naive thresholding, BH, BY, and a simultaneous post-hoc FDP certificate on one complete conformal anomaly family."
---

# FDR control and FDP certification

This example starts from one fixed test family and compares three decision
rules:

- unadjusted pointwise thresholding;
- Benjamini-Hochberg (BH), used by standard `select(...)`; and
- Benjamini-Yekutieli (BY), applied manually through SciPy.

It then constructs a simultaneous post-hoc upper bound for realized FDP over a
prespecified threshold grid.

## Complete example

```python
import numpy as np
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import conformal_fdp_upper_bound_from_result
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(800, 4))
x_family = np.vstack(
    [rng.normal(size=(95, 4)), rng.normal(loc=5.0, size=(5, 4))]
)
y_family = np.r_[np.zeros(95, dtype=int), np.ones(5, dtype=int)]
alpha = 0.1

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=100, random_state=42),
    strategy=Split(n_calib=0.4),
    seed=42,
).fit(x_reference)

bh_selected = np.asarray(detector.select(x_family, alpha=alpha))
result = detector.last_result
assert result is not None
assert result.p_values is not None
p_values = result.p_values

pointwise_selected = p_values <= alpha
by_adjusted = false_discovery_control(p_values, method="by")
by_selected = by_adjusted <= alpha

for name, selected in {
    "pointwise": pointwise_selected,
    "BH": bh_selected,
    "BY": by_selected,
}.items():
    print(
        name,
        {
            "discoveries": int(selected.sum()),
            "realized_fdp": float(false_discovery_rate(y_family, selected)),
            "power": float(statistical_power(y_family, selected)),
        },
    )

certificate = conformal_fdp_upper_bound_from_result(
    result,
    confidence=0.95,
    n_resamples=500,
    seed=42,
    thresholds=np.array([0.005, 0.01, 0.025, 0.05, 0.1]),
)
print(certificate.to_frame().to_string(index=False))
```

The pointwise rule does not account for the 100 simultaneous tests. BH is less
conservative than BY when its independence or positive-dependence conditions
apply. BY supports a broader dependence class, but neither procedure repairs
invalid p-values or adaptive family construction.

The labeled `realized_fdp` is the FDP of this one family. FDR is the expected
FDP over repetitions. Do not interpret one low realized FDP as validation of
the FDR theorem.

## Why the FDP certificate is different

The certificate is simultaneous over p-value thresholds within its documented
scope. This supports post-hoc threshold exploration while attaching a
high-confidence realized-FDP upper bound. It does not select by BH and does not
replace the expected-FDR target.

The result-based API accepts unweighted `Split` or detached calibration with
`Empirical` p-values. It rejects weighted, KDE, conditional-calibration, and
resampling-strategy result bundles.

The example uses 500 Monte Carlo resamples for speed. A reported analysis
should assess resampling stability and fix the envelope method before viewing
the final curve.

For weighted covariate shift, use the separate
[weighted WCS example](weighted_conformal.md). For online hypotheses and
single-stream change evidence, see the distinctions in the
[FDR guide](../user_guide/fdr_control.md#repeated-and-online-testing).
