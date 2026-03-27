# Common API Workflows

Task-first API map for common nonconform usage patterns.

## 1. Standard Conformal Detection

`select()` combines p-value computation and FDR-controlled selection in one
call — the recommended workflow for most use cases:

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(X_train)

mask = detector.select(X_test, alpha=0.05)
```

When you need the raw p-values for custom downstream analysis:

```python
p_values = detector.compute_p_values(X_test)
```

---

## 2. Scores-Only Workflow

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(X_train)

scores = detector.score_samples(X_test)
result = detector.last_result
```

Use this when you want calibrated scoring artifacts without applying FDR control yet.

---

## 3. Weighted Conformal + Weighted FDR Control

`select()` automatically dispatches to weighted FDR control when a
`weight_estimator` is configured:

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.enums import Pruning

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
detector.fit(X_train)

mask = detector.select(
    X_test,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)
```

---

## 4. Explicit Weight Preparation (Weighted Mode)

```python
detector.fit(X_train)
detector.prepare_weights_for(X_test_batch)
p_values = detector.compute_p_values(X_test_batch, refit_weights=False)
```

Use this when you need explicit state transitions for batched or exploratory workflows.

---

## 5. Pre-Trained Detector + Detached Calibration (Split Strategy)

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

# Train base detector in a separate step/pipeline
base_detector = IsolationForest(random_state=42)
base_detector.fit(X_fit)

# Attach pre-trained detector and calibrate on dedicated calibration data
detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.2),
    score_polarity="auto",
    seed=42,
)
detector.calibrate(X_calib)

p_values = detector.compute_p_values(X_test)
```

`calibrate(...)` is currently supported only for `Split` strategy.

---

## 6. Conditional Conformal P-Values + `select()`

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.scoring import ConditionalEmpirical

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    estimation=ConditionalEmpirical(method="simes", delta=0.1),
    score_polarity="auto",
    seed=42,
)
detector.fit(X_train)

mask = detector.select(X_test, alpha=0.05)
```

---

## 7. Temporal Session (Streaming Orchestration)

Use `TemporalSession` when you need one stateful call that combines:

- p-value generation (`ConformalDetector`)
- optional online FDR decisions (`controller.test_one(...)`)
- optional martingale monitoring

```python
from online_fdr.investing.alpha.alpha import Gai
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale
from nonconform.temporal import TemporalSession

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(X_train)

session = TemporalSession(
    detector=detector,
    online_controller=Gai(alpha=0.05, wealth=0.025),
    martingale=PowerMartingale(epsilon=0.5),
)

result = session.step(X_batch, apply_batch_select=True, alpha=0.05)
print(result.p_values.shape)
print(result.online_decisions.sum())
print(result.triggered_alarms)
```

`session.last_batch_decisions` stores the side-by-side batch BH/WCS mask from
the same p-values used in the step.

---

## Score Polarity

`ConformalDetector` expects anomaly-oriented scores internally (`higher = more anomalous`).

- `score_polarity="auto"`: infer known detector families.
- `score_polarity="higher_is_anomalous"`: no score transformation.
- `score_polarity="higher_is_normal"`: internally negates detector scores.

For unknown custom detectors in `auto`, nonconform raises a clear error and asks you
to set polarity explicitly.
