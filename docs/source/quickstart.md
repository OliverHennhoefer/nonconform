# Quick Start

Get started with nonconform in minutes.

## What You'll Learn

By the end of this guide, you'll know how to:

1. Wrap anomaly detectors to get statistically valid p-values
2. Use FDR control to make principled anomaly decisions
3. Use `detector.last_result` for downstream workflows
4. Choose between a minimal core path and an anomaly-ready path

**Prerequisites**: Familiarity with Python and basic anomaly detection concepts.

---

## Lane A: Core Quickstart (No Optional Extras)

This first runnable example uses only the core install (`pip install nonconform`).

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from scipy.stats import false_discovery_control

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

p_values = detector.compute_p_values(x_test)
discoveries = false_discovery_control(p_values, method="bh") < 0.05

print(f"Discoveries: {discoveries.sum()} / {len(x_test)}")
print(f"True anomalies in test set: {y_true.sum()}")
```

`score_polarity="auto"` handles sklearn score orientation automatically for
supported estimators.

---

## Lane B: Anomaly-Ready Quickstart (PyOD + oddball)

If you want benchmark datasets and a wider detector zoo immediately:

```bash
pip install "nonconform[pyod,data]"
```

```python
from oddball import Dataset, load
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

detector = ConformalDetector(
    detector=IForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(x_train)

p_values = detector.compute_p_values(x_test)
discoveries = false_discovery_control(p_values, method="bh") < 0.05
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
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split
from nonconform.fdr import weighted_false_discovery_control
from nonconform.enums import Pruning
from nonconform.metrics import false_discovery_rate, statistical_power

rng = np.random.default_rng(7)
x_train, _ = make_blobs(
    n_samples=900,
    centers=1,
    n_features=2,
    cluster_std=1.0,
    random_state=7,
)
x_test_normal, _ = make_blobs(
    n_samples=250,
    centers=1,
    n_features=2,
    cluster_std=1.2,
    random_state=8,
)
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

p_values = detector.compute_p_values(x_test)
result = detector.last_result  # ConformalResult bundle

discoveries = false_discovery_control(p_values, method="bh") < 0.05
weighted_discoveries = weighted_false_discovery_control(
    result=result,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=7,
)

print(f"BH discoveries: {discoveries.sum()}")
print(f"Weighted discoveries: {weighted_discoveries.sum()}")
print(f"FDR: {false_discovery_rate(y_true, discoveries):.3f}")
print(f"Power: {statistical_power(y_true, discoveries):.3f}")
```

---

## Next Steps

- **[Installation](installation.md)** - choose core vs anomaly-ready install profile
- **[Common API Workflows](api/common_workflows.md)** - task-first API map
- **[FDR Control](user_guide/fdr_control.md)** - detailed multiple testing guidance
- **[Weighted Conformal](user_guide/weighted_conformal.md)** - handling distribution shift
- **[Examples](examples/index.md)** - full end-to-end examples
