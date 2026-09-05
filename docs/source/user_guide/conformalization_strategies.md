---
description: "Understand how nonconform fits detectors, constructs calibration scores, and scores test data under each conformalization strategy."
---

# Conformalization strategies

A conformalization strategy determines three things:

1. which observations fit each base detector,
2. which observations produce calibration scores, and
3. which fitted models score a future test observation.

Those choices affect compute, memory, data efficiency, and the statistical
argument available for the resulting evidence. `DerandomizedSplits` constructs
e-values; the other strategies support p-value estimation. All use the public
`ConformalDetector.fit(...)` and `select(...)` workflow.

!!! important "Start with `Split` for the most direct validity argument"

    `Split` gives the most direct finite-sample argument in this package: the
    scoring rule is fitted without the calibration observations, then held
    fixed while calibration and test scores are compared. The CV and bootstrap
    strategies below are useful computational constructions, but their
    anomaly-score aggregation does not automatically inherit every coverage
    theorem proved for CV+, jackknife+, or JaB+ prediction intervals.

## Comparison at a glance

| Strategy | Model fits | Calibration scores | Models retained (plus mode where available) | Main tradeoff |
|---|---:|---|---:|---|
| `Split` | 1 | Held-out split | 1 | Clean separation, but reserves data for calibration |
| `DerandomizedSplits(R)` | `R` | Separate held-out scores per model | `R` | Averages e-values across splits; retains all models |
| `CrossValidation(k=k)` | `k` | One out-of-fold score per input row | `k` | Uses every row for calibration, with more fitting and inference work |
| `CrossValidation.jackknife()` | `n` | One leave-one-out score per input row | `n` | Maximum fit count and memory |
| `JackknifeBootstrap(B)` | `B` | Aggregated out-of-bag score per input row | `B` | Bootstrap-based calibration and configurable ensemble size |

Here, `n` is the number of rows passed to `fit(...)` and `B` is
`n_bootstraps`; `R` is `n_repetitions`.

## `Split`

`Split` randomly partitions the array passed to `fit(...)` into a proper
training subset and a calibration subset. It fits one detector on the former
and scores the latter with that fixed model. `seed` on `ConformalDetector`
controls the split and is propagated where supported.

```python
from nonconform import Split

proportional = Split(n_calib=0.2)
fixed_size = Split(n_calib=200)

print(proportional.calib_size)
print(fixed_size.calib_size)
```

`n_calib` accepts either:

- a float strictly between `0` and `1`, interpreted as a proportion; or
- an integer of at least `1` that leaves at least one row for fitting.

For a proportional split, the calibration count is rounded up. With classical
empirical p-values and `n_cal` calibration scores, the attainable values are
multiples of `1 / (n_cal + 1)`. Choose a calibration size that makes the p-value
resolution compatible with the downstream testing procedure.

## `DerandomizedSplits`

Use repeated integrated splits when dependence on a particular calibration
split matters. Each repetition fits a fresh model and retains its own
calibration scores. `select()` computes per-split conformal e-values, averages
them uniformly, and applies e-BH once to the fixed test batch.

```python
from nonconform import DerandomizedSplits

strategy = DerandomizedSplits(n_repetitions=5, n_calib=0.2)
print(strategy.get_params())
```

`n_calib` follows `Split` semantics and `n_repetitions` must be a positive
integer. One repetition is supported. The defaults (five repetitions and 10%
calibration) are starting points rather than universally optimal choices.
Fitting is sequential; `fit(n_jobs=...)` is unsupported for this strategy.

Configure advanced settings on the strategy: `alpha_bh=None` resolves to
`alpha / 10` during selection; a fixed explicit value must lie in `(0, 1)`.
Choose it before inspecting test evidence. `tie_seed=None` automatically
randomizes tied scores using a separate stream derived during fitting. An
integer overrides only the tie stream. A detector seed reproduces the fitting
and tie streams; without a seed, new randomness is generated once per fit.
Unchanged models and batches use the same tie seed on subsequent selections.

The resulting mask is available from `select()`, and evidence and diagnostics
from `last_selection_result`. `calibration_set` has shape
`(n_repetitions, n_calibration)` aligned with `detector_set`.
`score_samples()` still aggregates raw scores for inspection; its `aggregation`
setting never affects e-value selection. P-value methods and detached
`calibrate()` are unsupported. Non-identity weighting and non-Empirical
estimation configurations are rejected because this procedure constructs
e-values directly.

This retains all models in memory, unlike manually collecting score snapshots
and discarding models. It enables repeated inference without refitting, but does
not give joint FDR control over separately analyzed batches. See
[the e-value guarantee scope](fdr_control.md#derandomized-conformal-e-values)
and [the Shuttle example](../examples/derandomized_e_values.md).

## `CrossValidation`

`CrossValidation(k=...)` uses shuffled K-fold splitting by default. Each input
row receives one calibration score from a model that was not fitted on that
row.

```python
from nonconform import CrossValidation

cv_plus = CrossValidation(k=5)
cv_single_model = CrossValidation(k=5, mode="single_model")

print(cv_plus.k, cv_plus.mode)
print(cv_single_model.k, cv_single_model.mode)
```

In `mode="plus"`, which is the default, all fold models are retained. At
inference, each model scores every test row and `ConformalDetector` aggregates
those raw scores with its configured `aggregation` method. In
`mode="single_model"`, the fold models only construct out-of-fold calibration
scores; one additional model is then fitted on all input rows for inference.

!!! warning "`single_model` changes the calibration-to-test construction"

    In `single_model` mode, calibration scores come from fold models while test
    scores come from a different full-data model. It is cheaper at inference,
    but it has a weaker validity story. Do not describe it as equivalent to
    plus mode.

### Leave-one-out factory

`CrossValidation.jackknife()` sets the number of folds to the sample count at
fit time and disables shuffling. It therefore fits one leave-one-out model per
input row.

```python
from nonconform import CrossValidation

jackknife_plus = CrossValidation.jackknife()
jackknife_single_model = CrossValidation.jackknife(mode="single_model")

print(jackknife_plus.k, jackknife_plus.mode)
print(jackknife_single_model.k, jackknife_single_model.mode)
```

The factory reports `k is None` before fitting because the actual fold count is
the eventual sample count. Plus mode retains all leave-one-out models, so both
fit cost and inference cost grow linearly with `n` model evaluations.

## `JackknifeBootstrap`

`JackknifeBootstrap` fits `n_bootstraps` models on bootstrap resamples. Every
input row receives an out-of-bag calibration score aggregated across the
bootstrap models that did not include that row. The implementation constructs
resamples so that every row has out-of-bag coverage.

```python
from nonconform import JackknifeBootstrap

strategy = JackknifeBootstrap(
    n_bootstraps=100,
    aggregation_method="mean",
    mode="plus",
)

print(strategy.n_bootstraps)
print(strategy.aggregation_method)
```

`aggregation_method` controls aggregation of out-of-bag calibration scores and
accepts `"mean"` or `"median"`. It is distinct from
`ConformalDetector(aggregation=...)`, which combines retained models' test
scores.

Only `JackknifeBootstrap.fit_calibrate(...)` accepts `n_jobs`. Pass it through
the detector:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, JackknifeBootstrap

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(120, 3))

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=20, random_state=42),
    strategy=JackknifeBootstrap(n_bootstraps=5),
    seed=42,
)
detector.fit(x_reference, n_jobs=1)

print(len(detector.detector_set))
print(detector.calibration_set.shape)
```

Use more bootstrap iterations only when empirical stability justifies their
additional cost. The constructor requires at least two; there is no
distribution-free universal value that is adequate for every detector and
dataset.

## Plus and single-model modes

`CrossValidation` and `JackknifeBootstrap` accept the same two modes:

| Mode | Calibration construction | Test scoring | Cost profile |
|---|---|---|---|
| `"plus"` | Out-of-fold or out-of-bag | Aggregate all retained resampling models | Higher inference memory and latency |
| `"single_model"` | Out-of-fold or out-of-bag | One additional full-data model | Lower inference cost, weaker statistical alignment |

The name `plus` signals the package's resampling and multi-model scoring
construction. It should not be used as shorthand for a theorem without
checking that theorem's exact algorithm, target, and assumptions.

## Weighted mode

All strategies expose calibration rows for weight estimation, but `Split`
offers the clearest weighted covariate-shift workflow because its proper
training, calibration, and test roles remain explicit. If you combine weighted
inference with a resampling strategy, validate the exact construction and avoid
claiming the split weighted-conformal guarantee by analogy.

## What to compare empirically

When labels are available solely for evaluation, compare candidate strategies
on the same untouched evaluation family:

- realized false discovery proportion and statistical power;
- variability across repeated random seeds or resamples;
- fitting time, per-batch scoring time, and peak memory;
- calibration resolution and the number of discoveries;
- sensitivity to plausible distribution shifts.

Do not select a strategy on the same labeled evaluation set later used to
report final performance. That turns the evaluation set into tuning data.

## References and scope

- [Lei et al. (2018), *Distribution-Free Predictive Inference for Regression*](https://doi.org/10.1080/01621459.2017.1307116)
  develops split conformal prediction under exchangeability.
- [Barber et al. (2021), *Predictive Inference with the Jackknife+*](https://doi.org/10.1214/20-AOS1965)
  analyzes jackknife+ prediction intervals.
- [Kim, Xu, and Barber (2020), *Predictive Inference Is Free with the Jackknife+-after-Bootstrap*](https://proceedings.neurips.cc/paper/2020/hash/2b346a0aa375a07f5a90a344a61416c4-Abstract.html)
  analyzes JaB+ prediction intervals.
- [Vovk (2015), *Cross-conformal predictors*](https://doi.org/10.1007/s10472-013-9368-4)
  studies cross-conformal prediction.

These papers motivate important parts of the strategy design. Their prediction
set or interval results should not be quoted as direct proofs for every
anomaly-score aggregation implemented here.

Next, use [Choosing calibration strategies](choosing_strategies.md) to turn
these mechanics into a task-specific decision.
