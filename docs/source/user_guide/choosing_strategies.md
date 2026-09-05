---
description: "Choose a nonconform calibration strategy from the required validity claim, calibration resolution, data budget, and measured compute cost."
---

# Choosing calibration strategies

There is no dataset-size threshold at which one strategy becomes universally
best. Choose from the guarantee you need, then measure the statistical and
computational tradeoffs on data that represents deployment.

## Recommended decision order

### 1. State the validity claim

If you need the clearest finite-sample conformal argument, start with
`Split`. It separates proper training data from calibration data and applies a
fixed scoring rule to calibration and test observations.

Use `CrossValidation` or `JackknifeBootstrap` because their use of data or
their empirical stability is valuable for your task, not because a `+` suffix
automatically provides the same guarantee. The library aggregates anomaly
scores across fitted models; prediction-interval theorems for CV+,
jackknife+, and JaB+ do not transfer automatically to that construction.

If the goal is to combine repeated split evidence, use `DerandomizedSplits`.
It constructs e-values separately for each split and applies one final e-BH
selection through `select()`. This differs from raw-score aggregation in CV
and bootstrap strategies; see [FDR control](fdr_control.md#derandomized-conformal-e-values).

### 2. Check calibration resolution

With `Empirical(tie_break="classical")` and `n_cal` calibration scores,
p-values lie on the grid

$$
\left\{\frac{1}{n_{\mathrm{cal}}+1},
\frac{2}{n_{\mathrm{cal}}+1}, \ldots, 1\right\}.
$$

For `Split`, increasing the calibration share refines that grid but leaves less
data for fitting the anomaly detector. Cross-validation and bootstrap
strategies produce one calibration score per input row, but require more model
fits and have a different validity story.

!!! tip "Calculate before choosing"

    Write down the batch size, target `alpha`, intended multiple-testing
    procedure, and attainable p-value grid. A strategy can be statistically
    valid yet unable to make discoveries at the resolution your testing family
    requires.

### 3. Budget model fits and retained models

| Strategy | Fits during `fit(...)` | Retained models (plus mode where available) | Test scoring per row |
|---|---:|---:|---:|
| `Split` | 1 | 1 | 1 model |
| `DerandomizedSplits(R)` | `R` | `R` | `R` models |
| `CrossValidation(k=k)` | `k` | `k` | `k` models |
| `CrossValidation.jackknife()` | `n` | `n` | `n` models |
| `JackknifeBootstrap(B)` | `B` | `B` | `B` models |

For CV and bootstrap, `mode="single_model"` reduces retained models and test-time scoring, but adds a
full-data fit and changes the relationship between calibration and test scores.
Treat that as a statistical choice, not merely an optimization flag.

### 4. Account for the base detector

The strategy multiplies the cost and behavior of the detector you provide.
Measure the whole pipeline with realistic feature count, sample count, model
hyperparameters, and test batch size. A fold count or bootstrap count that is
cheap for one detector may be infeasible for another.

### 5. Compare on untouched evaluation data

When trustworthy labels exist, compare:

- realized false discovery proportion and power;
- variability across repeated seeds or resamples;
- fitting time, scoring latency, and peak memory;
- the number and stability of discoveries;
- behavior under deployment-relevant shifts.

Use one dataset for strategy selection and another for the final report. If the
same labeled data guides the choice and supplies the reported result, the final
estimate is optimistically selected.

## Sensible starting points

| Constraint | Starting point | Reason to reconsider |
|---|---|---|
| Strongest, easiest-to-audit validity story | `Split(n_calib=...)` | Calibration grid is too coarse or fitting data is too scarce |
| Reduce sensitivity to a random split with e-value aggregation | `DerandomizedSplits(n_repetitions=5, n_calib=...)` | Retained-model memory or repeated fitting cost is excessive |
| A moderate resampling budget | `CrossValidation(k=5, mode="plus")` | Fit or inference cost is excessive, or its validity scope is insufficient |
| Leave-one-out construction is specifically justified | `CrossValidation.jackknife(mode="plus")` | `n` fits and retained models are impractical |
| Bootstrap out-of-bag stability is specifically useful | `JackknifeBootstrap(..., mode="plus")` | Results are unstable across bootstrap counts or cost is excessive |
| Inference memory is the binding constraint | A `single_model` mode | The weaker calibration-to-test alignment is unacceptable |

These are starting configurations, not recommended sample-size regimes.

## Complete comparison example

This example compares three strategies on one labeled evaluation set. It is an
illustration of the measurement pattern, not evidence that the winning strategy
will generalize to another dataset.

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
x_reference = rng.normal(size=(240, 4))
x_evaluation = np.vstack(
    [
        rng.normal(size=(190, 4)),
        rng.normal(loc=4.0, size=(10, 4)),
    ]
)
y_evaluation = np.r_[np.zeros(190, dtype=int), np.ones(10, dtype=int)]

permutation = rng.permutation(len(x_evaluation))
x_evaluation = x_evaluation[permutation]
y_evaluation = y_evaluation[permutation]

strategies = {
    "split": Split(n_calib=0.3),
    "cv_plus": CrossValidation(k=5, mode="plus"),
    "bootstrap_plus": JackknifeBootstrap(n_bootstraps=10, mode="plus"),
}

for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=IsolationForest(
            n_estimators=30,
            max_samples=128,
            random_state=42,
        ),
        strategy=strategy,
        seed=42,
    )

    started = perf_counter()
    detector.fit(x_reference)
    fit_seconds = perf_counter() - started

    started = perf_counter()
    selected = np.asarray(detector.select(x_evaluation, alpha=0.1))
    score_seconds = perf_counter() - started

    print(
        name,
        {
            "discoveries": int(selected.sum()),
            "fdp": float(false_discovery_rate(y_evaluation, selected)),
            "power": float(statistical_power(y_evaluation, selected)),
            "fit_seconds": round(fit_seconds, 3),
            "score_seconds": round(score_seconds, 3),
        },
    )
```

The metric function reports realized false discovery proportion on this
particular labeled family, even though its API name is
`false_discovery_rate`. FDR is the expectation of that random proportion over
repetitions of the data-generating and selection process. One evaluation batch
cannot demonstrate FDR control by itself.

## Cases that require a different question

### Distribution shift

Resampling does not repair a calibration-to-test distribution shift. If the
shift is plausibly covariate shift and support overlaps, consider the
[weighted conformal workflow](weighted_conformal.md). If anomaly semantics or
the conditional mechanism changes, importance weighting alone is not a
solution.

### Sequential monitoring

Choosing among `Split`, `CrossValidation`, and `JackknifeBootstrap` does not
turn batch p-values into a conditionally valid sequential p-value process. For
change monitoring, use [exchangeability martingales](exchangeability_martingales.md)
and keep the scoring rule fixed during an episode.

### Labeled anomalies in training data

These conformalization strategies do not automatically clean contaminated
reference data. Define the null population and curate the proper training and
calibration data accordingly. Any data-dependent cleaning step becomes part of
the procedure whose validity must be assessed.

## Avoid these shortcuts

- Do not choose a strategy solely from labels such as “small” or “large” data.
- Do not claim that bootstrap resampling makes non-exchangeable data
  exchangeable.
- Do not assume that more folds or bootstraps monotonically improve power or
  validity.
- Do not tune the strategy on the final evaluation family.
- Do not compare strategies with different preprocessing, seeds, or test
  families unless that difference is intentional and reported.

For the exact implementation mechanics and literature scope, see
[Conformalization strategies](conformalization_strategies.md).
