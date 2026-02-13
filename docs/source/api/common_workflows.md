# Common API Workflows

Task-first API map for common nonconform usage patterns.

## 1. Standard Conformal Detection

```python
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(X_train)

p_values = detector.compute_p_values(X_test)
discoveries = false_discovery_control(p_values, method="bh") < 0.05
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

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.enums import Pruning
from nonconform.fdr import weighted_false_discovery_control

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
detector.fit(X_train)

_ = detector.compute_p_values(X_test)
discoveries = weighted_false_discovery_control(
    result=detector.last_result,
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

## Score Polarity

`ConformalDetector` expects anomaly-oriented scores internally (`higher = more anomalous`).

- `score_polarity="auto"`: infer known detector families.
- `score_polarity="higher_is_anomalous"`: no score transformation.
- `score_polarity="higher_is_normal"`: internally negates detector scores.

For unknown custom detectors in `auto`, nonconform raises a clear error and asks you
to set polarity explicitly.
