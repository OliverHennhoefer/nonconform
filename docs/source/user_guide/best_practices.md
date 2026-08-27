---
description: "Production practices for leakage-free conformal anomaly detection, batch FDR control, weighted inference, and sequential monitoring."
---

# Best practices

Reliable conformal anomaly detection is primarily a data-design problem. The
library can enforce API invariants, but it cannot infer the null population,
undo leakage, or decide which observations constitute one testing family.

## Begin with the statistical unit

Before fitting anything, write down:

- the **null case**: what counts as a normal observation;
- the **observation unit**: row, event, account, device, window, or another unit;
- the **batch family**: which hypotheses are selected together under one FDR
  target; or
- the **monitoring episode**: which ordered observations contribute to one
  sequential evidence process.

These definitions determine what exchangeability, FDR, and false-alarm control
mean. They should not be chosen after seeing the results.

!!! important "FDR control and sequential monitoring answer different questions"

    Batch FDR control limits the expected false discovery proportion within a
    declared multiple-testing family. A conformal martingale accumulates
    evidence over an ordered stream and can support an anytime false-alarm
    bound. Do not substitute one for the other merely because both consume
    p-values.

## Keep data roles separate

A strict split workflow has distinct roles:

| Role | May fit preprocessing? | May fit detector? | May calibrate p-values? | May evaluate reported performance? |
|---|---:|---:|---:|---:|
| Proper training | Yes | Yes | No | No |
| Calibration | No | No | Yes | No |
| Tuning/validation | No | No | No | No, if used to choose the final procedure |
| Final evaluation/deployment family | No | No | No | Yes |

When `ConformalDetector.fit(...)` uses `Split`, it creates the proper-training
and calibration roles internally. If you need explicit ownership of the split,
fit the base detector on proper training data and call `calibrate(...)` on a
separate calibration array.

### Put learned preprocessing inside the detector pipeline

This lets every strategy fit preprocessing only on the rows used to fit each
base detector. Do not standardize the entire reference array before a split,
because that leaks calibration information into the scoring rule.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(1_000, 5))
x_test = np.vstack(
    [rng.normal(size=(48, 5)), rng.normal(loc=4.0, size=(2, 5))]
)

base_detector = make_pipeline(
    StandardScaler(),
    IsolationForest(random_state=42),
).fit(x_reference[:700])
detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.3),
    score_polarity="higher_is_normal",
).calibrate(x_reference[700:])

selected = detector.select(x_test, alpha=0.1)
print(np.flatnonzero(selected))
```

Apply fixed, nonlearned transformations consistently at all stages. Treat any
transformation that estimates means, scales, encodings, components, feature
selection, imputation values, or thresholds as learned preprocessing.

## Curate the reference null

Calibration p-values describe extremeness relative to the calibration score
distribution. If the calibration set contains anomalies, the reference tail
can become heavier and genuine anomalies can receive larger p-values. If it
excludes legitimate normal subpopulations, those groups may be over-flagged.

There is no label-free cleaning rule that guarantees a valid null reference.
Use provenance, domain review, temporal boundaries, and held-out labeled data
where available. If a detector is used to clean its own calibration set,
include that adaptive cleaning step in the procedure you validate.

## Freeze the scoring construction

For a strict inductive workflow, the score map is fixed before it is applied to
calibration and test observations. This includes:

- detector parameters and fitted state;
- learned preprocessing;
- feature order and units;
- score polarity;
- any score aggregation rule.

Select hyperparameters on separate tuning data or by a protocol fixed in
advance. Reusing the final test family to tune a detector, choose a strategy,
or choose `alpha` invalidates an untouched evaluation.

## Choose calibration size from resolution and fit quality

With `n_cal` classical empirical calibration scores, the smallest p-value is
`1 / (n_cal + 1)`. Increasing the split calibration share refines the grid but
reduces the data available to fit the detector. Evaluate both sides of that
tradeoff.

Do not rely on universal rules such as “at least 1,000 calibration samples.”
The required resolution depends on the testing-family size, target level,
dependence procedure, desired power, and weight distribution. The detector's
fit requirements depend on its model class and feature geometry.

## Make batch families explicit

For one fixed family, call `select(...)` once on the complete batch whenever
practical:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(1_200, 4))
x_family = np.vstack(
    [rng.normal(size=(95, 4)), rng.normal(loc=4.5, size=(5, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

discoveries = detector.select(x_family, alpha=0.1)
print(f"{discoveries.sum()} discoveries in a family of {len(x_family)}")
```

Splitting one conceptual family into chunks and applying BH separately changes
the procedure. Each chunk may have its own per-family guarantee, but that is
not the same as FDR control over the original combined family. If memory forces
chunked score computation, collect all unweighted p-values and apply the chosen
multiple-testing procedure once to the complete family.

Weighted inference is batch-dependent because the test batch helps estimate
the density ratio and WCS uses the joint batch. Arbitrary chunking therefore
changes both weights and selection; it is not merely a memory optimization.

## Choose `alpha` from consequences

`alpha` is an error-budget decision. Choose it from the cost of investigations,
the harm of false alarms, the cost of missed anomalies, and any domain-specific
requirements. There is no generally correct “industry” value.

Record the chosen value and selection procedure before inspecting the family's
p-values. Trying several values and reporting the most attractive result is
another form of tuning.

## Evaluate what the guarantee targets

For one labeled family, the realized false discovery proportion is

$$
\operatorname{FDP}=\frac{V}{\max(R,1)},
$$

where `V` is the number of false discoveries and `R` the number of
discoveries. FDR is the expectation of FDP over repetitions. A single batch can
show its FDP and power, but cannot by itself demonstrate FDR control.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(800, 3))
x_evaluation = np.vstack(
    [rng.normal(size=(95, 3)), rng.normal(loc=4.0, size=(5, 3))]
)
y_evaluation = np.r_[np.zeros(95, dtype=int), np.ones(5, dtype=int)]

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)
selected = np.asarray(detector.select(x_evaluation, alpha=0.1))

print("realized FDP:", float(false_discovery_rate(y_evaluation, selected)))
print("power:", float(statistical_power(y_evaluation, selected)))
```

For empirical evidence about error control, repeat complete independently
generated or resampled experiments under a predeclared data-generating design.
Report the distribution of FDP, its average, power, discovery count, and the
fraction of runs with no discoveries. Do not average only successful runs.

## Treat weighted mode as a model, not a switch

Weighted conformal inference is appropriate when the calibration and target
covariate distributions differ, while the relevant conditional mechanism is
stable and the target distribution has support within the calibration
distribution. The weight estimator approximates a density ratio; it cannot
create missing support.

Before relying on weighted results:

- document why covariate shift is plausible;
- inspect whether calibration and test samples are nearly separable;
- examine weight distributions and clipping sensitivity;
- compare standard and weighted results on held-out shifted scenarios;
- use `select(...)` so weighted mode dispatches to WCS rather than ordinary BH;
- fit weights for the actual test family.

See [Weighted conformal inference](weighted_conformal.md) for a complete
workflow.

## Keep sequential episodes immutable

For a monitoring episode, fit the scorer once, prime rank history, and then
process observations in order. Do not refit the scorer, alter preprocessing, or
discard inconvenient observations mid-episode.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_fit = rng.normal(size=(400, 3))
x_prime = rng.normal(size=(200, 3))
x_stream = np.vstack(
    [rng.normal(size=(30, 3)), rng.normal(loc=3.0, size=(20, 3))]
)

monitor = ExchangeabilityMonitor(
    detector=IsolationForest(random_state=42),
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(ville_threshold=20.0)
    ),
    seed=42,
)
monitor.fit(x_fit).prime(x_prime)
states = monitor.update_many(x_stream)

first_alarm = next(
    (state for state in states if "ville" in state.triggered_alarms),
    None,
)
print(None if first_alarm is None else first_alarm.evidence_step)
```

Refitting begins a new statistical episode. If the system repeatedly resets or
runs several alarm rules and acts when any fires, allocate the lifetime error
budget across those opportunities. A per-episode Ville threshold does not
automatically provide a lifetime guarantee across unlimited restarts.

CUSUM and Shiryaev-Roberts thresholds are useful change-evidence triggers but
require separate calibration. They are not probability-of-ever-crossing bounds
solely because their inputs came from a martingale.

## Reproducibility and observability

- Set `seed` on `ConformalDetector` or `ExchangeabilityMonitor` when you need a
  reproducible stochastic construction.
- Record the package version, detector class and parameters, strategy and mode,
  estimator settings, feature schema, data provenance, family definition, and
  error target.
- Preserve discovery masks together with the corresponding p-values and
  `last_result` diagnostics.
- Time fitting and scoring in the deployment environment. Do not publish
  hardware-independent latency claims from an illustrative benchmark.
- Use structured application logs around data versions and episode boundaries;
  use the package logger and `verbose` controls only for library progress and
  diagnosis.

Randomized tie breaking and randomized WCS pruning consume random numbers. A
fixed seed makes a run reproducible, but does not remove the statistical role
of randomization.

## Production review checklist

### Before fitting

- Define the null population, batch family, or monitoring episode.
- Assign proper-training, calibration, tuning, and final-evaluation roles.
- Put learned preprocessing inside the fitted detector pipeline.
- Confirm that the detector exposes a fixed, pointwise `decision_function` and
  that score polarity is correct.

### Before selection or monitoring

- Confirm the feature schema and data-collection process match the protocol.
- Check attainable p-value resolution.
- State the FDR procedure and `alpha`, or the martingale and alarm thresholds.
- In weighted mode, assess covariate-shift plausibility and support overlap.
- In sequential mode, confirm the scorer will remain frozen for the episode.

### Before reporting

- Report realized FDP separately from expected FDR.
- Include power, discovery counts, variability, runtime, and no-discovery runs.
- Describe every adaptive choice made using labeled data.
- State assumptions and failure modes next to the claimed guarantee.
