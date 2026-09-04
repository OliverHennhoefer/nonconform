---
description: "Run nonconform's two primary workflows: FDR-targeted batch discovery and exchangeability-martingale change monitoring."
---

# Quick Start

Get started with nonconform in minutes.

This guide runs both primary workflows: batch discovery control and sequential
change monitoring. Both examples use only the core installation.

## What You'll Learn

By the end of this guide, you'll know how to:

1. Wrap an anomaly detector and calibrate its scores
2. Select discoveries from a fixed batch with FDR control
3. Monitor an ordered stream with sequential conformal p-values and a martingale
4. Inspect `detector.last_result` for downstream batch workflows
5. Add PyOD models and benchmark datasets when needed

**Prerequisites**: Familiarity with Python and basic anomaly detection concepts.

!!! note "Guarantee scope"

    The examples assume that the normal training/calibration data and normal
    test points are exchangeable. If deployment data is shifted, start with
    [Weighted Conformal](user_guide/weighted_conformal.md) before relying on
    the same batch validity claims. The sequential example additionally relies
    on a scoring rule fixed before monitoring and on the validity of its
    randomized sequential conformal p-values.

---

## Batch discovery control

This first runnable example uses only the core install (`pip install nonconform`).

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)

# Build a simple synthetic anomaly detection task
x_normal, _ = make_blobs(
    n_samples=1_200,
    centers=1,
    n_features=2,
    cluster_std=1.0,
    random_state=42,
)

x_train = x_normal[:800]  # normal-only training set
x_test_normal = x_normal[800:]
x_test_anomaly = rng.uniform(low=-8.0, high=8.0, size=(200, 2))
x_test = np.vstack([x_test_normal, x_test_anomaly])
y_true = np.hstack([
    np.zeros(len(x_test_normal), dtype=int),
    np.ones(len(x_test_anomaly), dtype=int),
])

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(x_train)

discoveries = detector.select(x_test, alpha=0.05)

print(f"Discoveries: {discoveries.sum()} / {len(x_test)}")
print(f"True anomalies in test set: {y_true.sum()}")
```

`score_polarity="auto"` handles sklearn score orientation automatically for
supported estimators.

---

## Sequential change monitoring

The fitted unweighted `Split` detector can initialize an
`ExchangeabilityMonitor`. Its held-out calibration scores become rank history;
martingale evidence starts at one and is updated only by stream observations.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_train = rng.normal(size=(1_000, 2))
x_stream = np.vstack([
    rng.normal(size=(50, 2)),
    rng.normal(loc=3.0, size=(50, 2)),
])

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
).fit(x_train)

monitor = ExchangeabilityMonitor.from_split_detector(
    detector,
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(restarted_ville_threshold=20.0)
    ),
    seed=42,
)

for x_t in x_stream:
    state = monitor.update(x_t)
    if "restarted_ville" in state.triggered_alarms:
        print(f"Change alarm at step {state.evidence_step}")
        break
else:
    print("No alarm in this finite stream")
```

For one valid null stream, a Ville threshold of `1 / alpha` bounds the
probability of ever crossing by `alpha`. CUSUM and Shiryaev-Roberts thresholds
are separate change-evidence triggers and need their own calibration. Continue
with [Exchangeability Martingales](user_guide/exchangeability_martingales.md)
before interpreting alarms operationally.

---

## PyOD and benchmark datasets (optional extras)

If you want benchmark datasets and a wider detector zoo immediately:

```bash
pip install "nonconform[pyod,data]"
```

```python
from oddball import Dataset, load
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

detector = ConformalDetector(
    detector=IForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(x_train)

discoveries = detector.select(x_test, alpha=0.05)
print(f"Discoveries: {discoveries.sum()}")
print(f"Anomaly rate in test set: {y_test.mean():.1%}")
```

---

## Loading Benchmark Datasets (Optional `[data]`)

For experimentation, use the `oddball` package:

```bash
pip install "nonconform[data]"
```

```python
from oddball import Dataset, load

x_train, x_test, y_test = load(Dataset.BREASTW, setup=True)
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Anomaly rate: {y_test.mean():.1%}")
```

---

## Evaluating Results and Accessing `last_result`

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(7)
x_normal, _ = make_blobs(
    n_samples=1_150,
    centers=1,
    n_features=2,
    cluster_std=1.0,
    random_state=7,
)
x_train = x_normal[:900]
x_test_normal = x_normal[900:]
x_test_anomaly = rng.uniform(low=-7.0, high=7.0, size=(80, 2))
x_test = np.vstack([x_test_normal, x_test_anomaly])
y_true = np.hstack([
    np.zeros(len(x_test_normal), dtype=int),
    np.ones(len(x_test_anomaly), dtype=int),
])

detector = ConformalDetector(
    detector=IsolationForest(random_state=7),
    strategy=Split(n_calib=0.25),
    score_polarity="auto",
    seed=7,
)
detector.fit(x_train)

discoveries = detector.select(x_test, alpha=0.05)
result = detector.last_result  # ConformalResult bundle from select()
p_values = result.p_values

print(f"Discoveries: {discoveries.sum()}")
print(f"P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
print(f"Realized FDP: {float(false_discovery_rate(y_true, discoveries)):.3f}")
print(f"Power: {float(statistical_power(y_true, discoveries)):.3f}")
```

---

## Next Steps

- **[Installation](installation.md)** - choose core vs anomaly-ready install profile
- **[Common API Workflows](api/common_workflows.md)** - task-first API map
- **[FDR Control](user_guide/fdr_control.md)** - detailed multiple testing guidance
- **[Exchangeability Martingales](user_guide/exchangeability_martingales.md)** - sequential change monitoring
- **[Weighted Conformal](user_guide/weighted_conformal.md)** - handling distribution shift
- **[Examples](examples/index.md)** - full end-to-end examples
