---
title: "nonconform: Conformal Anomaly Detection in Python"
description: "Python package for conformal anomaly detection with calibrated p-values, principled thresholds, and FDR-controlled decisions."
---

# nonconform: Conformal Anomaly Detection in Python

**Turn anomaly scores into calibrated decisions.**

Traditional anomaly detectors output scores and require arbitrary thresholds.
`nonconform` converts raw scores to conformal p-values and supports principled
False Discovery Rate (FDR) control for rigorous decision-making.

The short version for practitioners:

- Use `ConformalDetector.select(...)` when you need decisions, not just scores.
- Treat every guarantee as conditional on its assumptions: exchangeability for
  standard conformal workflows, and covariate-shift assumptions plus reliable
  weights for weighted workflows.
- Use the docs to check whether your data collection process matches those
  assumptions before relying on the error-control claims.

## The Problem

```python
# Traditional approach: arbitrary threshold, no formal error control
scores = detector.decision_function(X_test)
anomalies = scores > 0.5
```

## The Solution

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(),
    score_polarity="auto",
)
detector.fit(X_train)

discoveries = detector.select(X_test, alpha=0.05)
```

## When to Use nonconform

Use this library when you need:

- Statistical guarantees with clearly stated assumptions
- Principled thresholds instead of ad hoc cutoffs
- Multiple testing correction
- Calibrated uncertainty for downstream workflows

## Citation

If you use **nonconform** in academic work, reports, or other published
material, please cite the accompanying paper:

```bibtex
@misc{hennhöfer2026conformalanomalydetectionpython,
      title={Conformal Anomaly Detection in Python: Moving Beyond Heuristic Thresholds with 'nonconform'},
      author={Oliver Hennhöfer and Maximilian Kirsch and Christine Preisach},
      year={2026},
      eprint={2605.13642},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.13642},
}
```

## Guarantee Scope

nonconform does not make a weak detector "correct." It calibrates detector
scores against reference data. The calibration can control false positives only
when the reference data, test data, and selection procedure match the documented
assumptions.

| Workflow | Main assumption | Practical check |
|---|---|---|
| Standard conformal | Calibration and test points are exchangeable | Same population, same measurement process, no systematic time/order effect |
| FDR selection | Input p-values are valid and satisfy the method's dependence assumptions | Prefer `select(...)`; avoid unsupported p-value post-processing |
| Weighted conformal | Feature distribution may shift, but the anomaly mechanism is stable and supports overlap | Inspect shift, weights, and domain plausibility before trusting WCS results |

## Quick Links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [User Guide](user_guide/index.md)
- [Examples](examples/index.md)
- [Common API Workflows](api/common_workflows.md)
- [API Reference](api/index.md)

## Key Features

- Split conformal inference with finite-sample marginal guarantees
- Detector agnostic design (PyOD, scikit-learn, custom detectors)
- Multiple calibration strategies, including data-efficient resampling variants
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
