---
description: "Use scikit-learn, PyOD, and custom anomaly detectors while preserving score polarity and a fixed training-only score map."
---

# Detector compatibility

`ConformalDetector` can wrap scikit-learn estimators, PyOD models, and custom
detectors that satisfy a small interface. Statistical compatibility requires
more than having the right method names: the fitted detector must define a
fixed pointwise scoring rule for calibration and test observations.

## Required contract

A detector must:

- implement `fit(X, y=None)` and return itself;
- implement `decision_function(X)` and return one finite numeric score per row;
- implement `get_params(deep=True)` and `set_params(**params)`;
- support shallow and deep copying;
- retain all state needed to score new rows after fitting; and
- score each row without adapting the score map to the other rows in the
  evaluation batch.

The last condition is validity-critical. If `decision_function(X)` recomputes a
reference distribution from `X`, the score of one test point depends on the
test batch and no longer matches a strict training-only inductive construction.

!!! important "Internal score convention"

    `nonconform` normalizes every detector to **larger score means more
    anomalous** before calibration. A reversed polarity reverses the tail and
    invalidates the intended interpretation even though all returned numbers
    may look plausible.

## Scikit-learn

The following recognized scikit-learn estimators use larger
`decision_function` values for more normal observations, so `nonconform`
automatically negates their scores when polarity is omitted:

- `IsolationForest`
- `OneClassSVM`
- `SGDOneClassSVM`
- `LocalOutlierFactor`
- `EllipticEnvelope`

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(500, 3))
x_test = np.vstack(
    [rng.normal(size=(18, 3)), rng.normal(loc=4.0, size=(2, 3))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

print(detector.score_polarity)
print(detector.compute_p_values(x_test))
```

For `LocalOutlierFactor`, use `novelty=True`; otherwise scikit-learn does not
expose `decision_function` for unseen observations.

### Pipelines and unrecognized meta-estimators

Polarity inference is based on the outer estimator type. A scikit-learn
`Pipeline` around `IsolationForest`, for example, is not itself one of the
recognized normality-estimator classes. Set its polarity explicitly:

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(400, 4))
x_test = rng.normal(size=(10, 4))

pipeline = make_pipeline(
    StandardScaler(),
    IsolationForest(random_state=42),
)
detector = ConformalDetector(
    detector=pipeline,
    strategy=Split(n_calib=0.25),
    score_polarity="higher_is_normal",
    seed=42,
).fit(x_reference)

print(detector.compute_p_values(x_test))
```

Putting learned preprocessing inside the pipeline prevents it from being fitted
on held-out calibration rows.

## PyOD

Install PyOD support with:

```bash
pip install "nonconform[pyod]"
```

PyOD detectors conventionally expose larger `decision_function` scores for
more anomalous observations, and `nonconform` resolves that polarity
automatically.

```python
import numpy as np
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(500, 3))
x_test = np.vstack(
    [rng.normal(size=(18, 3)), rng.normal(loc=4.0, size=(2, 3))]
)

detector = ConformalDetector(
    detector=IForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

print(detector.compute_p_values(x_test))
```

### Hard-blocked batch-adaptive detectors

`nonconform` rejects `CD`, `COF`, `COPOD`, `ECOD`, `LMDD`, `LOCI`, `RGraph`,
`SOD`, and `SOS` because their evaluation scoring does not provide the fixed
training-only score map required by the strict inductive workflow.

The block is based on class identity/name, not a claim that every unblocked
PyOD detector has been theoretically certified. Verify unfamiliar or newly
released detectors against the required contract.

Meta-estimators such as `FeatureBagging`, `LSCP`, and `SUOD` inherit the
behavior of their component detectors. They require explicit inspection;
`SUOD`, for example, can include blocked detector families in a default or
custom base-estimator list.

## Custom detector

The following complete detector uses squared standardized distance from the
fitted feature means. It is intentionally simple, but it demonstrates the
entire protocol without pseudocode.

```python
from typing import Any, Self

import numpy as np

from nonconform import ConformalDetector, Split

class StandardizedDistanceDetector:
    def __init__(self, variance_floor: float = 1e-12) -> None:
        self.variance_floor = variance_floor

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Self:
        del y
        X = np.asarray(X, dtype=float)
        self.location_ = X.mean(axis=0)
        self.scale_ = np.maximum(X.std(axis=0), self.variance_floor)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        standardized = (X - self.location_) / self.scale_
        return np.square(standardized).sum(axis=1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {"variance_floor": self.variance_floor}

    def set_params(self, **params: Any) -> Self:
        for name, value in params.items():
            if name != "variance_floor":
                raise ValueError(f"Unknown parameter: {name}")
            self.variance_floor = value
        return self

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(400, 3))
x_test = np.vstack(
    [rng.normal(size=(18, 3)), rng.normal(loc=4.0, size=(2, 3))]
)

detector = ConformalDetector(
    detector=StandardizedDistanceDetector(),
    strategy=Split(n_calib=0.3),
    score_polarity="higher_is_anomalous",
    seed=42,
).fit(x_reference)

print(detector.compute_p_values(x_test))
```

For custom estimators, omitted polarity defaults to
`"higher_is_anomalous"`. Prefer setting it explicitly so the convention is
reviewable. Explicit `score_polarity="auto"` is deliberately strict and raises
for a custom class whose family is unknown.

## Automatic parameter normalization

During construction and fitting, `nonconform` inspects `get_params()` and may
set common parameters:

| Parameter aliases | Value | Purpose |
|---|---|---|
| `contamination` | Smallest positive Python float | Prevent detector-level contamination thresholds from defining the conformal decision rule |
| `n_jobs`, `n_threads`, or `num_workers` | `-1` | Use available parallel workers when the detector exposes such a parameter |
| `random_state`, `seed`, or `random_seed` | Detector seed | Reproducible fitting where supported |

Absent contamination and parallelism parameters are acceptable. If a seed is
provided but the detector exposes no recognized seed parameter, the library
warns because it cannot guarantee reproducibility for a stochastic detector.

`ConformalDetector` owns copied detector state. Custom objects containing file
handles, sessions, native resources, or other noncopyable state need an
appropriate `__copy__`/`__deepcopy__` implementation or a simpler serializable
configuration object.

## Compatibility review for an unfamiliar detector

Before using a new detector, verify all of the following from its implementation
or authoritative documentation:

1. `decision_function` is available for unseen rows after `fit`.
2. It returns exactly one numeric value per row.
3. The score direction is known.
4. Test scoring does not refit, renormalize against the test batch, or use test
   neighbors in a transductive way.
5. Repeated scoring of the same row with a frozen model is stable, apart from
   explicitly modeled independent randomness.
6. Parameters and fitted state can be copied for the selected strategy.
7. Learned preprocessing is included inside the fitted object or otherwise
   trained without leakage.

Method presence answers only items 1 and 2. The remaining items determine
whether conformal calibration has the interpretation you intend.

## References

- [scikit-learn novelty and outlier detection guide](https://scikit-learn.org/stable/modules/outlier_detection.html)
  documents score conventions and `LocalOutlierFactor` novelty behavior.
- [PyOD documentation](https://pyod.readthedocs.io/)
  documents the common PyOD detector API and model catalog.
- [Custom detector example in this repository](https://github.com/OliverHennhoefer/nonconform/blob/main/examples/custom/centroid_detector.py)
  provides another complete implementation.
