---
description: "Derandomized conformal e-value example for aggregating repeated split-conformal anomaly evidence with FDR control."
---

# Derandomized Conformal E-Values

Use conformal e-values to aggregate evidence from repeated random
training/calibration splits and apply one final e-BH decision.
`DerandomizedSplits` manages the repetitions through the familiar
`detector.fit(...)` and `detector.select(...)` workflow.

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
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, DerandomizedSplits, Split
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

# Fit all replicas once; select() constructs e-values and applies e-BH.
detector = ConformalDetector(
    detector=IForest(),
    strategy=DerandomizedSplits(n_repetitions=5, n_calib=n_calib),
    score_polarity="higher_is_anomalous",
    seed=42,
).fit(x_train)
e_mask = detector.select(x_test, alpha=alpha)
e_result = detector.last_selection_result
assert e_result is not None

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
- `e_result.tie_seed` records the effective randomized tie seed.

The default inner threshold is `alpha_bh=alpha / 10`, following the paper's
practical recommendation. It must be fixed before inspecting results.
The inner cutoff uses an inclusive comparison with `alpha_bh`.

The strategy handles tied scores automatically, using a random stream separate
from fitting. Set `seed` on the detector to reproduce the splits and ties, or
supply `tie_seed=...` on the strategy to override only tie handling. With
`seed=None`, new randomness is generated once per fit; selecting the same batch
again uses the same tie seed, assuming deterministic model scoring.

`last_selection_result` is a defensive snapshot. Raw scoring, refitting, or
reconfiguration clears it. `last_result` remains reserved for raw-score and
p-value snapshots; it is `None` after e-value selection. `score_samples()` still
returns aggregated raw scores, but those scores are not used for e-value
selection. Its snapshot has `calib_scores=None` because there is no single
calibration distribution for the aggregated raw scores.

The strategy retains every fitted model, increasing inference memory compared
with collecting score snapshots manually. `calibration_set` is a matrix with
one calibration-score row per model. P-value methods and detached calibration
are unsupported; use `fit()` and `select()`. Weighting and non-Empirical
estimation configurations are rejected. Defaults are starting points, not
universally optimal settings.

## Advanced: select from existing split results

The standalone API remains available when you already manage individual splits:

```python
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import select_conformal_e_values

x_reference, x_test, _ = load(Dataset.SHUTTLE, setup=True, seed=1)
results = []
for seed in range(5):
    detector = ConformalDetector(
        detector=IForest(), strategy=Split(n_calib=1_000), seed=seed,
    ).fit(x_reference)
    detector.score_samples(x_test)
    result = detector.last_result
    assert result is not None
    results.append(result)

selection = select_conformal_e_values(results, alpha=0.2, tie_seed=2026)
print("discoveries:", selection.selected.sum())
```

Unlike the strategy, this standalone function rejects ties by default;
`tie_seed=...` explicitly enables reproducible tie randomization. The example
uses different fitting seeds from the high-level example, so its numerical
results need not match.

Pass unmodified snapshots. The standalone result-aware API checks integrated,
unweighted `Split` provenance and exact test-batch content and row order. It
rejects changed or reordered batches but does not track edits to score arrays.

## Guarantee Scope

`DerandomizedSplits` scores the same batch with every model internally. Neither
API can establish the scientific exchangeability assumptions from data alone.

The method additionally requires:

- the same fixed test family in the same row order across repetitions,
- exchangeable normal reference and null test observations,
- a strict score ordering or independently randomized secondary ranks,
- a fixed inner threshold and one final e-BH application.

The construction supplies an aggregate null-evidence condition sufficient for
e-BH; it does not claim that every constructed value is individually an
ordinary e-value. This implementation uses uniform aggregation only.

## Runnable Notebook

The notebook at `examples/derandomized_e_values.ipynb` also compares classical
Split + BH, CV-style raw-score aggregation + BH, and DerandomizedSplits + e-BH
on the same Shuttle test family, with a side-by-side table of realized results:

```bash
jupyter notebook examples/derandomized_e_values.ipynb
```
