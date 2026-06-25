---
description: "Derandomized conformal e-value example for stable repeated split-conformal anomaly decisions with FDR control."
---

# Derandomized Conformal E-Values

Use conformal e-values when ordinary split conformal decisions are too sensitive
to the random training/calibration split. The workflow repeats split conformal
scoring several times, aggregates the resulting e-values, and applies e-BH for
batch FDR control.

This example implements the uniform-aggregation path from Bashari et al.,
[Derandomized Novelty Detection with FDR Control via Conformal E-values](https://arxiv.org/abs/2302.07294);
the authors' reference code is available at
[Meshiba/derandomized-novelty-detection](https://github.com/Meshiba/derandomized-novelty-detection).

This is an expert workflow. For the default path, use `detector.select(...)`.

## Setup

```python
import numpy as np
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import conformal_e_value_selection
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)
```

!!! note "Prerequisites"
    This example uses PyOD and oddball:

    ```bash
    pip install "nonconform[pyod,data]"
    ```

## Baseline Split Selection

A single split conformal run is fast and simple, but the exact selected points
can depend on the calibration split.

```python
baseline = ConformalDetector(
    detector=IForest(random_state=1),
    strategy=Split(n_calib=1_000),
    seed=1,
)
baseline.fit(x_train)
baseline_mask = baseline.select(x_test, alpha=0.2)

print(f"Baseline discoveries: {baseline_mask.sum()}")
print(f"Baseline FDR: {false_discovery_rate(y_test, baseline_mask):.3f}")
print(f"Baseline power: {statistical_power(y_test, baseline_mask):.3f}")
```

## Repeated Split Scores

For e-values, collect raw test and calibration scores from repeated split runs.
`score_samples(...)` populates `detector.last_result` with `test_scores` and
`calib_scores` without applying a p-value or FDR decision layer.

```python
alpha = 0.2
test_scores = []
calib_scores = []

for seed in range(5):
    detector = ConformalDetector(
        detector=IForest(random_state=seed),
        strategy=Split(n_calib=1_000),
        seed=seed,
    )
    detector.fit(x_train)
    detector.score_samples(x_test)

    result = detector.last_result
    test_scores.append(result.test_scores)
    calib_scores.append(result.calib_scores)

test_scores = np.vstack(test_scores)
calib_scores = np.vstack(calib_scores)
```

## E-Value Selection

`conformal_e_value_selection(...)` computes conformal e-values for each split,
averages them uniformly, and applies e-BH.

```python
e_result = conformal_e_value_selection(
    test_scores,
    calib_scores,
    alpha=alpha,
)

e_mask = e_result.selected

print(f"Derandomized discoveries: {e_mask.sum()}")
print(f"Derandomized FDR: {false_discovery_rate(y_test, e_mask):.3f}")
print(f"Derandomized power: {statistical_power(y_test, e_mask):.3f}")
print(f"Inner alpha_bh: {e_result.alpha_bh:.3f}")
```

## Interpreting The Result

The output is a stable batch decision rule:

- `e_result.e_values`: aggregated evidence values; larger means stronger anomaly evidence.
- `e_result.selected`: final e-BH discoveries.
- `e_result.e_threshold`: selected e-value cutoff, or `inf` if nothing was selected.
- `e_result.n_repetitions`: number of split conformal runs that were aggregated.

The default `alpha_bh` is `alpha / 10`, following the paper's conservative
recommendation. Pass `alpha_bh=...` explicitly if you have a task-specific
reason to tune the inner threshold.

## Guarantee Scope

This workflow targets batch FDR control under the conformal e-value assumptions:

- normal training/calibration samples and null test samples are exchangeable,
- the score convention is fixed so larger scores mean more anomalous,
- each repeated analysis is a valid split-conformal score map,
- e-BH is applied once to the aggregated e-values.

E-values can be more stable than a single random split, but they can also be
more conservative than BH on p-values, especially when signals are weak or few
discoveries are expected.

## Runnable Notebook

The same workflow is available as a root notebook example:

```bash
jupyter notebook examples/derandomized_e_values.ipynb
```
