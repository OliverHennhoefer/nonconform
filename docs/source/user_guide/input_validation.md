---
description: "Validate nonconform parameters, feature matrices, calibration resolution, weighted batches, and fitted state without relying on arbitrary sample-size rules."
---

# Input validation and edge cases

This page distinguishes API constraints enforced by `nonconform` from
statistical adequacy, which depends on the task. Passing validation means that
an operation is well formed. It does not establish exchangeability, p-value
validity, adequate power, or reliable importance weights.

## Parameter reference

### Conformal detector

| Parameter | Accepted values | Notes |
|---|---|---|
| `aggregation` | `"mean"`, `"median"`, `"minimum"`, `"maximum"` | Case and surrounding whitespace are normalized; aggregation is across raw test scores from retained models |
| `score_polarity` | `None`, `"auto"`, `"higher_is_anomalous"`, `"higher_is_normal"`, or the corresponding enum | Explicitly configure custom detectors when the direction cannot be inferred safely |
| `seed` | `None` or a nonnegative integer | Controls supported stochastic components and deterministic seed derivation |
| `verbose` | `True` or `False` | Controls aggregation progress; resampling and weighting progress also depend on logger levels |
| `verify_prepared_batch_content` | `True` or `False` | When true, prepared weighted state is tied to the exact batch bytes, shape, and dtype |
| `select(..., alpha=...)` | float strictly between `0` and `1` | Target level for the selected FDR procedure |

The detector must be fitted with `fit(...)` or detached calibration must be
completed with `calibrate(...)` before scoring.

### `Split`

`Split(n_calib=...)` accepts:

- a float strictly between `0` and `1`, interpreted as the calibration
  proportion; or
- an integer at least `1`, interpreted as the calibration count.

At fit time, the resulting calibration count must be smaller than the number of
input rows so that at least one row remains for fitting the base detector. A
proportional count is rounded up.

```python
import math

from nonconform import Split

n_rows = 101
strategy = Split(n_calib=0.2)
n_calibration = math.ceil(n_rows * strategy.calib_size)

print(n_calibration)
```

### Cross-validation and bootstrap strategies

| Constructor | Constraint |
|---|---|
| `CrossValidation(k=k)` | `k` must be at least `2` and no greater than the number of rows at fit time |
| `CrossValidation.jackknife()` | Uses `k=n` at fit time and requires at least two rows |
| `CrossValidation(..., shuffle=...)` | `shuffle` must be Boolean |
| `JackknifeBootstrap(n_bootstraps=B)` | `B` must be an integer of at least `2` |
| Either resampling strategy's `mode` | `"plus"` or `"single_model"` |
| `JackknifeBootstrap(..., aggregation_method=...)` | `"mean"` or `"median"` |
| `fit(..., n_jobs=...)` with `JackknifeBootstrap` | `None`, `-1`, or a positive integer |

Passing `n_jobs` to `fit(...)` with a strategy that does not expose it raises a
`ValueError` instead of silently ignoring it.

### P-value estimation

| Estimator | Key constraints |
|---|---|
| `Empirical(tie_break=...)` | `"classical"` or `"randomized"` |
| `ConditionalEmpirical(delta=...)` | `0 < delta < 1` |
| `ConditionalEmpirical(method=...)` | `"mc"`, `"simes"`, `"dkwm"`, or `"asymptotic"` |
| `ConditionalEmpirical(simes_kden=...)` | Positive integer |
| `ConditionalEmpirical(mc_num_simulations=...)` | Integer of at least `100` |
| `Probabilistic(...)` | Requires the `probabilistic` extra and has model-specific parameter constraints |

`ConditionalEmpirical` rejects weighted p-values. `delta` configures its
conditional calibration map and is distinct from the selection target
`alpha`.

### Weight estimators

For `logistic_weight_estimator(...)` and `forest_weight_estimator(...)`,
`clip_quantile` must be `None` or strictly between `0` and `0.5`. The estimator
requires nonempty calibration samples and a test batch. Custom scikit-learn
classifiers used by `SklearnWeightEstimator` must implement `predict_proba`.

Weights supplied to the low-level weighted p-value function must be finite and
nonnegative, must have lengths matching their score arrays, and calibration
weights must have a positive sum.

!!! warning "Valid numbers can still be invalid evidence"

    Finite, nonnegative weights satisfy numerical validation, but that does not
    show that they approximate the required density ratio. Inspect overlap,
    classifier behavior, clipping sensitivity, and downstream calibration as
    described in [Weighted conformal inference](weighted_conformal.md).

## Feature matrix requirements

For the main fitted workflow, provide a two-dimensional numeric NumPy array or
pandas `DataFrame` with shape `(n_samples, n_features)`. Use the same feature
count, order, units, encoding, and preprocessing at fitting, calibration, and
inference.

```python
import numpy as np

x_reference = np.asarray(
    [
        [0.1, 1.2, -0.3],
        [0.0, 0.8, -0.1],
        [0.2, 1.0, -0.4],
    ],
    dtype=float,
)

if x_reference.ndim != 2:
    raise ValueError("x_reference must be two-dimensional")
if x_reference.shape[0] < 2 or x_reference.shape[1] < 1:
    raise ValueError("x_reference needs rows and features")
if not np.isfinite(x_reference).all():
    raise ValueError("x_reference must contain only finite values")

print(x_reference.shape)
```

This explicit check is useful at application boundaries because different base
detectors enforce dtype, missing-value, and infinity rules differently.
`nonconform` adapts pandas inputs but does not promise to repair invalid feature
data.

At inference, a pandas `Series` is interpreted as a batch with one feature, not
as one row with many features. Pass a one-dimensional NumPy array to
`compute_p_value(...)` for one observation, or a one-row `DataFrame`/2D array
to the batch methods.

## Calibration resolution

For classical unweighted empirical p-values,

$$
p(x)=\frac{1+\sum_{i=1}^{n_{\mathrm{cal}}}
\mathbf{1}\{S_i\ge S(x)\}}{n_{\mathrm{cal}}+1}.
$$

The smallest possible p-value is `1 / (n_cal + 1)`. This is a mathematical
resolution limit, not a reason to impose an arbitrary universal minimum
calibration size.

```python
import numpy as np

from nonconform import Empirical

calibration_scores = np.arange(10, dtype=float)
test_scores = np.array([-1.0, 4.5, 20.0])

p_values = Empirical().compute_p_values(test_scores, calibration_scores)
print(p_values)
print("minimum attainable:", 1 / (len(calibration_scores) + 1))
```

Choose `n_cal` by considering the testing-family size and procedure, desired
power, detector fitting needs, compute budget, and empirical stability. A
calibration set can be large enough for fine resolution and still be
unrepresentative of the deployment null.

### Ties

Classical empirical p-values count calibration scores tied with the test score
as at least as extreme. This is deterministic and conservative. Randomized tie
handling interpolates tied mass and requires a seed for reproducibility.

```python
import numpy as np

from nonconform import Empirical

calibration_scores = np.ones(20)
test_scores = np.ones(5)

classical = Empirical().compute_p_values(test_scores, calibration_scores)

randomized_estimator = Empirical(tie_break="randomized")
randomized_estimator.set_seed(42)
randomized = randomized_estimator.compute_p_values(
    test_scores,
    calibration_scores,
)

print("classical:", classical)
print("randomized:", randomized)
```

Randomized p-values are valid only with the prescribed independent
randomization. They are not deterministic scores with finer-looking decimal
places.

## Fitted state and cached results

The following methods require a fitted or detached-calibrated detector:
`score_samples(...)`, `compute_p_value(...)`, `compute_p_values(...)`,
`select(...)`, and `prepare_weights_for(...)`.

After a scoring call, `last_result` is either `None` or a defensive
`ConformalResult` copy. Its fields depend on the call:

| Latest call | `p_values` | scores | weights |
|---|---|---|---|
| `compute_p_values(...)` | Present | Present | Present in weighted mode |
| `select(...)` | Present | Present | Present in weighted mode |
| `score_samples(...)` | `None` | Present | Present in weighted mode |

The public `calibration_set`, `calibration_samples`, `detector_set`, and
`last_result` accessors return copies. Modifying a retrieved array does not
modify detector state.

## Prepared weighted batches

`prepare_weights_for(batch)` fits weight state for one concrete batch.
`compute_p_values(batch, refit_weights=False)` and
`select(batch, refit_weights=False)` require that preparation first.

With the default `verify_prepared_batch_content=True`, both batch length and
content must match. With verification disabled, only batch length is checked;
that option trades safety for lower hashing cost and should be used only when
the caller enforces identity.

## Assumptions cannot be tested into existence

Marginal feature tests, classifier accuracy, and drift metrics can reveal
differences between datasets. They cannot prove joint exchangeability or the
covariate-shift condition. Use them as diagnostics alongside knowledge of how
the data was sampled, transformed, and ordered.

Likewise, checking that p-values lie in `[0, 1]` detects numerical corruption,
not miscalibration. Calibration diagnostics require null data that was not used
to fit, calibrate, tune, or choose the displayed result.

## Common exceptions

| Exception | Typical cause | Resolution |
|---|---|---|
| `NotFittedError` | Scoring or preparing weights before fitting/calibration | Complete `fit(...)` or the supported detached `calibrate(...)` workflow |
| `ValueError` about `n_calib` | Invalid type/range or no proper-training rows remain | Reduce the calibration count or add reference rows |
| `ValueError` about folds | `k < 2` or `k > n` | Choose a valid fold count for the actual fit array |
| `ValueError` about `n_jobs` | Parallelism passed to an unsupported strategy | Pass it only with `JackknifeBootstrap` |
| `RuntimeError` about weights | Weighted state was not prepared, or weighted mode is disabled | Use `refit_weights=True` or prepare the exact batch |
| `ValueError` about prepared batch | Size or content differs from the prepared batch | Prepare weights again for the batch being scored |
| `ValueError` about feature count | Inference dimensionality differs from fitted dimensionality | Reproduce the fitted feature schema and preprocessing |

Do not branch production logic on complete exception strings. Catch the narrow
documented exception class where recovery is safe, and preserve the original
message for diagnosis.

## Pre-flight checklist

- Define the null population and the unit of exchangeability.
- Keep detector fitting, calibration, tuning, and final evaluation roles
  separate.
- Confirm feature schema and score polarity.
- Calculate the classical p-value grid for the actual calibration count.
- Declare each multiple-testing family before viewing its p-values.
- Treat weight quality and support overlap as statistical requirements, not
  mere input validation.
- Freeze the scoring rule during a sequential monitoring episode.
