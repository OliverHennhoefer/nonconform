---
description: "Diagnose nonconform imports, fitted state, score polarity, p-value resolution, FDR selection, weighted batches, sequential monitoring, and performance."
---

# Troubleshooting

Start from the first layer that fails: installation, detector interface,
fitting, score construction, p-values, selection, or guarantee assumptions.
Changing `alpha` before identifying the layer can hide the symptom without
fixing the cause.

## Import and installation errors

### A symbol is not available from `nonconform`

The package root intentionally exports the most common API only:

```python
from nonconform import (
    ConformalDetector,
    CrossValidation,
    Empirical,
    JackknifeBootstrap,
    Probabilistic,
    Split,
    forest_weight_estimator,
    logistic_weight_estimator,
)
```

Use documented module imports for specialized APIs:

```python
from nonconform.fdr import Pruning, conformal_fdp_upper_bound_from_result
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor
from nonconform.scoring import ConditionalEmpirical

print(Pruning.DETERMINISTIC)
print(AlarmConfig())
```

Do not import from `nonconform._internal`; that namespace is private and not a
compatibility contract.

### An optional dependency is missing

Install only the extra used by the failing workflow:

```bash
pip install "nonconform[pyod]"          # PyOD detectors
pip install "nonconform[data]"          # oddball datasets and generators
pip install "nonconform[fdr]"           # online_fdr package
pip install "nonconform[probabilistic]" # KDEpy and Optuna
pip install "nonconform[all]"           # every optional extra
```

Sequential monitoring and the built-in martingales are part of the core
installation and do not require the `fdr` extra.

## Detector construction errors

### Missing protocol methods

The base detector must implement `fit`, `decision_function`, `get_params`, and
`set_params`, and it must be copyable. A `predict` method alone is insufficient
because conformal calibration needs continuous scores.

See [Detector compatibility](detector_compatibility.md) for a complete custom
implementation.

### A PyOD detector is blocked

`CD`, `COF`, `COPOD`, `ECOD`, `LMDD`, `LOCI`, `RGraph`, `SOD`, and `SOS` are
rejected because their evaluation scoring is batch-adaptive rather than a
fixed training-only map. Choose a detector that scores unseen rows from frozen
fitted state. An unblocked class is not automatically certified; inspect its
behavior.

### Automatic score polarity fails

Explicit `score_polarity="auto"` recognizes PyOD and a small set of
scikit-learn normality estimators. For a custom detector or outer meta-estimator,
set the known convention explicitly:

- `"higher_is_anomalous"` if larger values are more unusual;
- `"higher_is_normal"` if larger values are more normal.

A scikit-learn pipeline around `IsolationForest` needs
`score_polarity="higher_is_normal"` because polarity inference sees the outer
pipeline type.

## Fitted-state and shape errors

### `NotFittedError`

Call `fit(...)` for integrated training and calibration, or use the supported
detached workflow: fit a base detector, construct `ConformalDetector` with
`Split`, and call `calibrate(...)` on separate data.

`score_samples`, `compute_p_value`, `compute_p_values`, `select`, and
`prepare_weights_for` all require fitted/calibrated state.

### Feature count differs

Use the same feature count, order, units, encoding, and learned preprocessing at
every stage. Prefer a fitted scikit-learn pipeline so transformations travel
with the detector.

At inference, a pandas `Series` is interpreted as a batch with one feature. For
one multifeature observation, use `compute_p_value(one_dimensional_numpy_row)`
or pass a one-row 2D array/DataFrame to a batch method.

### `n_calib`, folds, or bootstrap parameters fail

- `Split` needs at least one calibration row and at least one proper-training
  row.
- `CrossValidation` needs `2 <= k <= n` at fit time.
- `JackknifeBootstrap` needs at least two bootstrap iterations and at least two
  input rows.
- `fit(..., n_jobs=...)` is supported only by `JackknifeBootstrap`.

See [Input validation](input_validation.md) for the complete constraints.

## There is no `predict(...)` method

This is intentional. Choose the output that matches the task:

| Need | Method |
|---|---|
| Raw detector-scale values | `score_samples(...)` |
| One unweighted p-value | `compute_p_value(...)` |
| Batch p-values | `compute_p_values(...)` |
| FDR-controlled batch decisions | `select(...)` |
| Sequential evidence state | `ExchangeabilityMonitor.update(...)` |

Do not threshold raw scores and describe the result as conformal or
FDR-controlled.

## P-values look wrong

### Outside `[0, 1]` or nonfinite

This indicates invalid detector output, nonfinite feature data, or a custom
estimator defect. Inspect feature matrices and raw scores before changing any
statistical parameters.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(400, 3))
x_test = rng.normal(size=(20, 3))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

scores = np.asarray(detector.score_samples(x_test))
p_values = np.asarray(detector.compute_p_values(x_test))

assert np.isfinite(scores).all()
assert np.isfinite(p_values).all()
assert ((0.0 <= p_values) & (p_values <= 1.0)).all()
print(np.column_stack([scores, p_values])[:5])
```

This example intentionally calls `compute_p_values(...)` last because each
scoring method replaces `last_result`.

### P-values are coarse

With `n_cal` classical empirical calibration scores, values move in steps of
`1 / (n_cal + 1)`. This is expected. Increase calibration resolution only after
checking the tradeoff with detector-fitting data and compute.

Randomized tie breaking removes the positive grid floor through prescribed
randomization. `Probabilistic` produces continuous KDE estimates but changes
the validity story to a model-based one.

### P-values are reversed

Verify score polarity. Calibration and obvious synthetic anomalies should be
inspected on the normalized `score_samples(...)` scale; larger normalized
scores should correspond to more anomalous observations. This is a diagnostic,
not a substitute for validation on representative data.

## No discoveries

Check, in order:

1. **Detector signal:** do known held-out anomalies tend to receive larger
   normalized scores?
2. **Calibration grid:** is the smallest attainable p-value compatible with
   the family size and selection thresholds?
3. **Family size:** BH thresholds depend on the complete number of tests.
4. **Reference contamination:** anomalies in calibration can thicken the score
   tail.
5. **Conditional or weighted conservativeness:** inspect transformed p-values,
   weights, and pruning rather than assuming failure.
6. **Actual signal:** a valid procedure is allowed to return no discoveries.

Do not raise `alpha` solely until discoveries appear and then report the nominal
guarantee as if that target had been chosen in advance.

## Empirical FDP is too high

First confirm that labels use `1` for anomaly and `0` for normal and that the
metric receives `(y_true, selected_mask)`.

Then investigate:

- leakage from calibration/test data into fitting or preprocessing;
- calibration-to-test shift;
- invalid detector batch adaptation;
- score polarity;
- a mismatch between the declared family and evaluated pooled results;
- dependence outside the multiple-testing procedure's assumptions;
- adaptive model, method, or threshold selection on the evaluation data; and
- Monte Carlo variability from too few independently repeated families.

Lowering `alpha` may reduce discoveries but does not repair an invalid
construction.

## Weighted-mode problems

### Weights were not prepared

`refit_weights=False` requires a preceding
`prepare_weights_for(the_exact_batch)`. The default verifier checks size and
content. Prepare again if either changed.

### Weights are concentrated or unstable

Inspect `last_result.calib_weights` and `last_result.test_weights`, quantiles,
effective sample size, and sensitivity to the weight estimator and
`clip_quantile`. There is no universal acceptable weight range.

Near-perfect calibration/target discrimination often signals weak support
overlap. Clipping can stabilize computation but cannot create missing support.

### BH and weighted p-values disagree with `select(...)`

That is expected. Weighted `select(...)` uses WCS; ordinary BH is not the
documented weighted selection procedure.

See [Weighted conformal inference](weighted_conformal.md).

## Memory or scoring is slow

Determine which cost dominates:

- `Split` retains one detector.
- `CrossValidation(k, mode="plus")` retains `k` detectors.
- jackknife plus mode retains one detector per reference row.
- `JackknifeBootstrap(B, mode="plus")` retains `B` detectors.
- weighted inference fits a domain classifier for each target batch by default.
- exact sequential ranks retain the full score history and use linear-time
  sorted-list insertion.

For one large **unweighted** family, you may score chunks, collect the p-values,
and apply one multiple-testing procedure to the complete vector. This preserves
the family definition when detector scoring is pointwise and deterministic:

```python
import numpy as np
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(500, 3))
x_family = rng.normal(size=(120, 3))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

p_value_chunks = [
    np.asarray(detector.compute_p_values(x_family[start : start + 25]))
    for start in range(0, len(x_family), 25)
]
p_values = np.concatenate(p_value_chunks)
selected = false_discovery_control(p_values, method="bh") <= 0.05

print(selected.shape)
```

Do not use this pattern with randomized tie breaking, because separate calls
consume randomization differently from one batch call. Do not use it in
weighted mode, where the batch defines weight estimation and WCS coupling.

Measure real fit and inference paths before changing strategies. Hardware-free
latency promises and universal dataset-size cutoffs are not reliable guidance.

## Sequential monitoring problems

### A fixed-calibration loop is being treated as Ville-valid

Repeated `compute_p_value(...)` calls share a fixed calibration ECDF and are
only marginally valid by default. Use `ExchangeabilityMonitor` for randomized
sequential ranks.

### The scorer is refitted during an episode

Stop the episode. Refit, prime a new monitor, and account for the new testing
opportunity in the lifetime false-alarm budget.

### CUSUM or Shiryaev-Roberts threshold is called a false-alarm probability

Those thresholds are change-evidence triggers that need separate calibration.
Only the documented Ville thresholds have the probability-of-ever-crossing
interpretation under the valid e-process assumptions.

### Alarm labels are compared to point labels

A change alarm targets a stopping time, not row-level classification. Evaluate
null false-alarm probability, detection probability, delay, and censoring.

See [Streaming evaluation](streaming_evaluation.md) and
[Exchangeability martingales](exchangeability_martingales.md).

## Logging and progress

Set `logging.getLogger("nonconform").setLevel(...)` for package progress and
warnings. `ConformalDetector(verbose=True)` separately enables aggregation
progress. Exact logger names and progress labels are in
[Logging and progress](logging.md).

## Reporting a reproducible issue

Include:

- `nonconform`, Python, NumPy, SciPy, scikit-learn, and optional dependency
  versions;
- operating system;
- detector, strategy, estimation, weighting, polarity, and seed settings;
- input shapes and dtypes without sensitive data;
- complete traceback;
- the smallest runnable reproducer; and
- whether the issue concerns runtime behavior or a statistical claim.

Open an issue at the
[project tracker](https://github.com/OliverHennhoefer/nonconform/issues).
