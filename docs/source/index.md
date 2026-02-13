# nonconform

**Turn anomaly scores into statistically valid decisions.**

Traditional anomaly detectors output scores and require arbitrary thresholds.
nonconform converts raw scores to conformal p-values and supports principled
False Discovery Rate (FDR) control for final decisions.

## The Problem

```python
# Traditional approach: arbitrary threshold, no formal error control
scores = detector.decision_function(X_test)
anomalies = scores > 0.5
```

## The Solution

```python
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(),
    score_polarity="auto",
)
detector.fit(X_train)

p_values = detector.compute_p_values(X_test)
discoveries = false_discovery_control(p_values, method="bh") < 0.05
```

## When to Use nonconform

Use this library when you need:

- Statistical guarantees on anomaly decisions
- Principled thresholds instead of ad hoc cutoffs
- Multiple testing correction
- Calibrated uncertainty for downstream workflows

## Quick Links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [User Guide](user_guide/index.md)
- [Examples](examples/index.md)
- [Common API Workflows](api/common_workflows.md)
- [API Reference](api/index.md)

## Key Features

- Conformal inference with finite-sample guarantees
- Detector agnostic design (PyOD, scikit-learn, custom detectors)
- Multiple calibration strategies (Split, CV, Jackknife+ variants)
- FDR and weighted FDR workflows
- Covariate-shift handling via weighted conformal methods

## Installation

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

Ready to start? Continue with the [Quick Start guide](quickstart.md).
