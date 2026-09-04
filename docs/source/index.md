---
title: "nonconform: Conformal Anomaly Detection"
description: "Calibrate anomaly scores, control batch discoveries, and monitor streams for change with conformal methods in Python."
---

<p align="center">
  <img src="assets/banner.png" alt="nonconform" width="720">
</p>

# nonconform: Conformal Anomaly Detection in Python

**Calibrate scores. Control discoveries. Monitor change.**

`nonconform` turns anomaly scores into conformal evidence for two primary
workflows:

- **Batch discovery control:** compute conformal p-values and select anomalies
  with false discovery rate (FDR) control.
- **Sequential change monitoring:** transform a stream into randomized
  sequential conformal p-values and accumulate evidence against exchangeability
  with conformal martingales.

Both workflows can wrap supported scikit-learn estimators, PyOD models, or a
custom detector that implements the documented protocol.

## Batch discovery control

Use `select(...)` when a fixed batch of observations must become anomaly
decisions. The example below needs only the core installation.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_train = rng.normal(size=(1_000, 2))
x_test = np.vstack([
    rng.normal(size=(200, 2)),
    rng.normal(loc=5.0, size=(20, 2)),
])

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
).fit(x_train)

discoveries = detector.select(x_test, alpha=0.05)
p_values = detector.last_result.p_values

print(f"Selected {discoveries.sum()} of {len(x_test)} observations")
print(f"Smallest p-value: {p_values.min():.4f}")
```

`alpha=0.05` is the target FDR level for this batch, not an anomaly-score
threshold and not a promise about the realized false discovery proportion in
this particular run.

## Sequential change monitoring

Use `ExchangeabilityMonitor` when observations arrive in order and the goal is
to accumulate evidence that the stream has stopped being exchangeable with its
reference history.

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

A Ville threshold of 20 bounds the probability of ever crossing that threshold
by 0.05 on one valid null stream. It does not control FDR across streams.

## Guarantee scope

!!! important "Guarantees are assumption-dependent"

    Standard conformal workflows require the calibration data and null test
    cases to be exchangeable relative to a scoring rule fixed without using the
    calibration or test outcomes. BH selection additionally requires valid
    p-values and its dependence conditions. Weighted workflows require the
    stated covariate-shift model, support overlap, and reliable importance
    weights. Sequential Ville guarantees require conditionally valid sequential
    conformal p-values.

    `nonconform` calibrates detector scores. It cannot make an unsuitable
    detector, contaminated reference set, adaptive analysis, or mismatched data
    collection process valid.

| Workflow | Start here | Main output |
|---|---|---|
| Fixed batch of anomaly candidates | [Quick Start](quickstart.md#batch-discovery-control) | Conformal p-values and an FDR-controlled Boolean mask |
| Ordered stream monitored for change | [Exchangeability Martingales](user_guide/exchangeability_martingales.md) | Sequential p-values, e-values, evidence statistics, and configured alarms |
| Covariate shift between calibration and test | [Weighted Conformal](user_guide/weighted_conformal.md) | Weighted p-values and WCS selections |
| Custom or third-party detector | [Detector Compatibility](user_guide/detector_compatibility.md) | A validated, anomaly-oriented score interface |

## Installation

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

See [Installation](installation.md) for optional detector, dataset,
probabilistic-estimation, and online-FDR extras.

## Documentation map

- [Quick Start](quickstart.md) provides complete first examples.
- [Statistical Concepts](user_guide/statistical_concepts.md) defines the claims
  and assumptions used throughout the site.
- [Common API Workflows](api/common_workflows.md) maps tasks to public calls.
- [User Guide](user_guide/index.md) covers calibration strategies, weighting,
  FDR, monitoring, validation, and production practice.
- [API Reference](api/index.md) documents the complete public module surface.

## Citation

If you use `nonconform` in academic work, cite the accompanying paper:

```bibtex
@misc{hennhoefer2026,
  title={Conformal Anomaly Detection in Python: Moving Beyond Heuristic Thresholds with 'nonconform'},
  author={Oliver Hennhöfer and Maximilian Kirsch and Christine Preisach},
  year={2026},
  eprint={2605.13642},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2605.13642},
}
```
