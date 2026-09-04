---
description: "Compare conditionally calibrated empirical conformal p-value maps in one complete, reproducible anomaly-selection example."
---

# Conditional conformal selection

`ConditionalEmpirical` transforms ordinary empirical conformal p-values toward
a calibration-set-conditional validity target. This example compares all four
implemented maps on the same synthetic data.

## Complete comparison

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power
from nonconform.scoring import ConditionalEmpirical

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(700, 4))
x_test = np.vstack(
    [rng.normal(size=(15, 4)), rng.normal(loc=4.5, size=(5, 4))]
)
y_test = np.r_[np.zeros(15, dtype=int), np.ones(5, dtype=int)]

for method in ["mc", "simes", "dkwm", "asymptotic"]:
    estimation_kwargs = {
        "method": method,
        "delta": 0.1,
        "tie_break": "classical",
    }
    if method == "mc":
        estimation_kwargs["mc_num_simulations"] = 500

    detector = ConformalDetector(
        detector=IsolationForest(n_estimators=50, random_state=42),
        strategy=Split(n_calib=0.3),
        estimation=ConditionalEmpirical(**estimation_kwargs),
        seed=42,
    ).fit(x_reference)

    selected = np.asarray(detector.select(x_test, alpha=0.1))
    result = detector.last_result
    assert result is not None
    assert result.p_values is not None

    print(
        method,
        {
            "minimum_p": float(result.p_values.min()),
            "discoveries": int(selected.sum()),
            "realized_fdp": float(false_discovery_rate(y_test, selected)),
            "power": float(statistical_power(y_test, selected)),
        },
    )
```

The comparison is descriptive. Selecting the method that looks best on this
same labeled family and then reporting its metrics would be adaptive reuse of
the evaluation data. Use separate tuning and final-evaluation data for a method
comparison.

`delta=0.1` configures the conditional-calibration event; `alpha=0.1`
configures downstream selection. They happen to be equal here but control
different quantities and need not match.

The `"mc"` example uses only 500 simulations to keep the example quick. For a
reported analysis, assess Monte Carlo stability and choose the simulation count
before viewing the final family.

See [Conformal inference](../user_guide/conformal_inference.md#conditionally-calibrated-p-values)
for method scope, small-calibration fallback behavior, and the reference paper.
