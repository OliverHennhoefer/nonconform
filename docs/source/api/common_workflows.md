# Common workflows

This page collects complete, independent examples for the most common
`nonconform` tasks. Every code block includes its own imports and data setup, so
you can copy and run any example by itself.

## Batch discovery control

Use `select(...)` when a batch of observations forms one multiple-testing
family and you want a Boolean mask of discoveries. In standard, unweighted
mode, `select(...)` computes conformal p-values and applies the
Benjamini-Hochberg (BH) procedure at the requested false discovery rate (FDR)
level.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(800, 4))
x_test = np.vstack(
    [
        rng.normal(size=(18, 4)),
        rng.normal(loc=5.0, size=(2, 4)),
    ]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
)
detector.fit(x_reference)
selected = detector.select(x_test, alpha=0.05)

print(np.flatnonzero(selected))
print(f"discoveries: {selected.sum()}")
```

`alpha=0.05` is a target FDR level, not a statement that every selected
observation has a 95% probability of being anomalous. The FDR guarantee is a
property of the selection procedure under its assumptions and across the
declared testing family.

## Inspect p-values and scores

Call `compute_p_values(...)` when you need the p-values themselves. The
`last_result` snapshot also contains the test scores and calibration scores used
for that call.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(600, 4))
x_test = np.vstack(
    [rng.normal(size=(8, 4)), rng.normal(loc=5.0, size=(2, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

p_values = detector.compute_p_values(x_test)
result = detector.last_result
assert result is not None
assert result.p_values is not None

print(np.column_stack([result.test_scores, p_values]))
```

!!! note "`last_result` describes the latest scoring call"

    Calling `score_samples(...)`, `compute_p_value(...)`,
    `compute_p_values(...)`, or `select(...)` replaces the cached result.
    Retrieve `last_result` after the call you want to inspect. The returned
    object is a defensive copy.

## Detached calibration for a pre-fitted detector

Use `calibrate(...)` with `Split` when the base detector has already been fitted
on a proper training set and you have a separate calibration set. The scoring
rule must remain fixed while calibration and test observations are scored.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_fit = rng.normal(size=(400, 4))
x_calibration = rng.normal(size=(200, 4))
x_test = np.vstack(
    [rng.normal(size=(18, 4)), rng.normal(loc=5.0, size=(2, 4))]
)

base_detector = IsolationForest(random_state=42).fit(x_fit)
detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.3),
    seed=42,
)
detector.calibrate(x_calibration)

p_values = detector.compute_p_values(x_test)
print(p_values)
```

`n_calib` is not used to split data in this detached workflow, because the
entire array passed to `calibrate(...)` is the calibration set. It remains a
required `Split` constructor argument for consistency with the strategy API.

## Weighted conformal selection under covariate shift

Weighted mode estimates a target-to-calibration density ratio and uses weighted
conformal p-values. `select(...)` then applies weighted conformalized selection
(WCS), not ordinary BH.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(2_000, 3))
x_test = np.vstack(
    [
        rng.normal(loc=0.5, size=(18, 3)),
        rng.normal(loc=6.0, size=(2, 3)),
    ]
)

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.4),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
).fit(x_reference)

selected = detector.select(x_test, alpha=0.1)
result = detector.last_result
assert result is not None
assert result.calib_weights is not None
assert result.test_weights is not None

print("selected indices:", np.flatnonzero(selected))
print(
    "weight ranges:",
    (result.calib_weights.min(), result.calib_weights.max()),
    (result.test_weights.min(), result.test_weights.max()),
)
```

This workflow requires more than a classifier that separates calibration and
test samples. Its validity relies on an appropriate covariate-shift model,
support overlap, and sufficiently reliable density-ratio estimates. See
[Weighted conformal inference](../user_guide/weighted_conformal.md) for the
assumptions and diagnostics.

### Reuse explicitly prepared weights

Weight fitting is batch-specific. If you want that state transition to be
explicit, prepare weights for the exact batch and disable refitting on the
subsequent call.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(800, 3))
x_test = rng.normal(loc=0.6, size=(40, 3))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
).fit(x_reference)

detector.prepare_weights_for(x_test)
p_values = detector.compute_p_values(x_test, refit_weights=False)
print(p_values[:5])
```

By default, `nonconform` verifies both the size and content of the prepared
batch. Preparing weights for one batch and applying them to another is an
error.

## Conditionally calibrated conformal p-values

`ConditionalEmpirical` applies one of the package's conditional calibration
maps to classical empirical conformal p-values. The available maps include
finite-sample and asymptotic options. The class is available from
`nonconform.scoring` and is intentionally unsupported in weighted mode.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.scoring import ConditionalEmpirical

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(600, 4))
x_test = np.vstack(
    [rng.normal(size=(18, 4)), rng.normal(loc=5.0, size=(2, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    estimation=ConditionalEmpirical(delta=0.05, method="dkwm"),
    seed=42,
).fit(x_reference)

p_values = detector.compute_p_values(x_test)
print(p_values)
```

See [Conformal inference](../user_guide/conformal_inference.md) before choosing
a conditional calibration method. `delta` is the failure probability associated
with the calibration map; it is not the FDR target `alpha`.

## Sequential change monitoring

For a stream, use randomized sequential conformal ranks and a conformal
martingale. The following example fits a split conformal detector, freezes its
fitted scoring rule, primes rank history with the calibration scores, and then
processes stream observations in order.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(600, 3))
x_stream = np.vstack(
    [
        rng.normal(size=(30, 3)),
        rng.normal(loc=3.0, size=(20, 3)),
    ]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

monitor = ExchangeabilityMonitor.from_split_detector(
    detector,
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(ville_threshold=20.0)
    ),
    seed=42,
)
states = monitor.update_many(x_stream)
first_alarm = next((state for state in states if state.triggered_alarms), None)
if first_alarm is None:
    print("No alarm in this finite stream")
else:
    print(first_alarm.evidence_step, first_alarm.triggered_alarms)
```

!!! important "Batch p-values are not sequential conformal p-values"

    Do not feed repeated calls to `ConformalDetector.compute_p_value(...)` into
    a martingale and infer sequential validity. `ExchangeabilityMonitor`
    constructs the required sequential randomized ranks. Its null guarantee
    requires an exchangeable score sequence conditional on the frozen scoring
    rule. Do not refit the scorer during a monitoring episode.

At a Ville threshold of `1 / alpha`, a valid nonnegative martingale that starts
at one has probability at most `alpha` of ever crossing the threshold under
the null. CUSUM and Shiryaev-Roberts statistics are also exposed, but their
thresholds are change-evidence triggers and need separate calibration; they do
not automatically inherit the Ville guarantee. See
[Exchangeability martingales](../user_guide/exchangeability_martingales.md).

## Score direction for custom detectors

`nonconform` normalizes scores internally so that larger values mean more
anomalous observations. Built-in adapters resolve the convention for supported
scikit-learn and PyOD detectors. For a custom detector, specify the convention
explicitly if it cannot be inferred safely:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(300, 2))
x_test = rng.normal(size=(5, 2))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="higher_is_normal",
    seed=42,
).fit(x_reference)

print(detector.compute_p_values(x_test))
```

The explicit setting in this example matches scikit-learn's
`IsolationForest.decision_function`, where lower values are more abnormal.
