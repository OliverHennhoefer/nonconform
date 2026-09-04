---
description: "Run a complete split-conformal anomaly detection workflow with benchmark data, p-values, BH selection, and labeled evaluation."
---

# Classical split-conformal detection

This example loads a normal-only reference sample and a labeled test family,
fits a PyOD detector through `ConformalDetector`, and selects discoveries with
BH FDR control.

```bash
pip install "nonconform[data,pyod]"
```

## Complete example

```python
import numpy as np
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

x_reference, x_test, y_test = load(Dataset.SATIMAGE2, setup=True, seed=42)

detector = ConformalDetector(
    detector=IForest(n_estimators=100, random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

selected = np.asarray(detector.select(x_test, alpha=0.1))
result = detector.last_result
assert result is not None
assert result.p_values is not None

print("reference rows:", len(x_reference))
print("test rows:", len(x_test))
print("true anomalies:", int(np.asarray(y_test).sum()))
print("discoveries:", int(selected.sum()))
print("smallest p-values:", np.sort(result.p_values)[:5])
print("realized FDP:", float(false_discovery_rate(y_test, selected)))
print("power:", float(statistical_power(y_test, selected)))
```

The reference sample returned by `setup=True` contains normal observations for
fitting and calibration. The test labels are used only after selection for
evaluation.

`select(...)` computes p-values and decisions in one pass. Retrieve
`last_result` immediately afterward to inspect the same p-values; another
scoring call replaces the cached snapshot.

!!! note "The displayed FDP is not the FDR guarantee"

    `false_discovery_rate(...)` computes the realized false discovery
    proportion for this one labeled family. FDR is its expectation over
    repetitions of the complete procedure. One benchmark family can measure
    FDP and power, but cannot demonstrate FDR control by itself.

## What to vary responsibly

- Change the base detector only using separate tuning data.
- Change `n_calib` after checking both detector-fitting needs and the p-value
  grid `1 / (n_cal + 1)`.
- Choose `alpha` from the operational error budget before inspecting the test
  p-values.
- Repeat the full evaluation across prespecified datasets or seeds and report
  variability, including runs with no discoveries.

For resampling alternatives, continue with
[Data-efficient resampling](resampling_conformal.md). For the exact BH
assumptions, see [FDR control](../user_guide/fdr_control.md).
