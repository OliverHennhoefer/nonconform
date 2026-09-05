---
description: "Derandomized conformal e-value example for aggregating repeated split-conformal anomaly evidence with FDR control."
---

# Derandomized Conformal E-Values

Use conformal e-values to aggregate evidence from repeated random
training/calibration splits and apply one final e-BH decision. This is an expert
alternative when sensitivity to one split matters; the default batch workflow
remains `detector.select(...)`.

The example implements uniform aggregation from Bashari et al.,
[Derandomized Novelty Detection with FDR Control via Conformal E-values](https://arxiv.org/abs/2302.07294).

!!! note "Prerequisites"
    This example uses PyOD and oddball:

    ```bash
    pip install "nonconform[pyod,data]"
    ```

## Complete Example

The entire Python block is independently runnable. It keeps the test family and
row order fixed across repetitions and uses a seeded random secondary ordering
for any tied scores.

```python
import numpy as np
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import select_conformal_e_values
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)
alpha = 0.2
n_calib = 1_000

# Single-split reference workflow.
baseline = ConformalDetector(
    detector=IForest(random_state=1),
    strategy=Split(n_calib=n_calib),
    score_polarity="higher_is_anomalous",
    seed=1,
).fit(x_train)
baseline_mask = baseline.select(x_test, alpha=alpha)

# Repeat valid integrated Split constructions for the same test family.
results = []
for split_seed in range(5):
    detector = ConformalDetector(
        detector=IForest(random_state=split_seed),
        strategy=Split(n_calib=n_calib),
        score_polarity="higher_is_anomalous",
        seed=split_seed,
    ).fit(x_train)
    detector.score_samples(x_test)

    result = detector.last_result
    assert result is not None
    assert result.test_scores is not None
    assert result.calib_scores is not None
    results.append(result)

e_result = select_conformal_e_values(
    results,
    alpha=alpha,
    tie_seed=2026,
)
e_mask = e_result.selected

print(f"Baseline discoveries: {baseline_mask.sum()}")
print(
    "Baseline realized FDP: "
    f"{false_discovery_rate(y=y_test, y_hat=baseline_mask):.3f}"
)
print(
    "Baseline realized true-positive rate: "
    f"{statistical_power(y=y_test, y_hat=baseline_mask):.3f}"
)
print(f"Derandomized discoveries: {e_mask.sum()}")
print(
    "Derandomized realized FDP: "
    f"{false_discovery_rate(y=y_test, y_hat=e_mask):.3f}"
)
print(
    "Derandomized realized true-positive rate: "
    f"{statistical_power(y=y_test, y_hat=e_mask):.3f}"
)
print(f"Inner alpha_bh: {e_result.alpha_bh:.3f}")
```

The displayed FDP and true-positive rate describe this one realized labeled
batch. They do not estimate repeated-sampling FDR or prove that aggregation is
more stable in a particular application.

## Interpreting the Result

- `e_result.e_values` contains the uniformly aggregated evidence values.
- `e_result.selected` is the final e-BH discovery mask.
- `e_result.e_threshold` is the selected e-value cutoff, or `inf` if empty.
- `e_result.n_repetitions` and `e_result.n_calibration` record input sizes.
- `e_result.tie_seed` records the randomized tie seed, or `None` when ties are
  rejected.

The default inner threshold is `alpha_bh=alpha / 10`, following the paper's
practical recommendation. It must be fixed before inspecting results.
The inner cutoff uses an inclusive comparison with `alpha_bh`.

## Guarantee Scope

Pass unmodified result snapshots. The result-aware API checks their recorded
integrated, unweighted `Split` provenance and exact test-batch content and row
order. Results scored on changed or reordered batches are rejected. Edits to
snapshot score arrays are not tracked, and the API cannot establish the
scientific exchangeability assumptions.

The method additionally requires:

- the same fixed test family in the same row order across repetitions,
- exchangeable normal reference and null test observations,
- a strict score ordering or independently randomized secondary ranks,
- a fixed inner threshold and one final e-BH application.

The construction supplies an aggregate null-evidence condition sufficient for
e-BH; it does not claim that every constructed value is individually an
ordinary e-value. This implementation uses uniform aggregation only.

## Runnable Notebook

The equivalent notebook is available at `examples/derandomized_e_values.ipynb`:

```bash
jupyter notebook examples/derandomized_e_values.ipynb
```
