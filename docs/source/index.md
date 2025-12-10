# nonconform

**Turn anomaly scores into statistically valid decisions.**

Traditional anomaly detectors output scores and require you to pick an arbitrary threshold. How do you know if a score of -0.5 really means "anomaly"? nonconform solves this by converting raw scores into **p-values**—probabilities that tell you how likely an observation is to be normal. Combined with **False Discovery Rate (FDR) control**, you get principled anomaly detection with mathematical guarantees on your error rates.

## The Problem

```python
# Traditional approach - arbitrary thresholds, no statistical guarantees
scores = detector.decision_function(X_test)
anomalies = scores > 0.5  # Why 0.5? What's my false positive rate?
```

## The Solution

```python
from nonconform import ConformalDetector, Split
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

# Wrap any anomaly detector with nonconform
detector = ConformalDetector(IForest(), Split())
detector.fit(X_train)

# Get p-values instead of arbitrary scores
p_values = detector.predict(X_test)

# Control false discovery rate at 5%
decisions = false_discovery_control(p_values, method='bh') < 0.05
```

With nonconform, you know that among all the points you flag as anomalies, at most 5% will be false positives—guaranteed by statistical theory, not guesswork.

## When to Use nonconform

Use this library when you need:

- **Statistical guarantees** on your anomaly detection results
- **Principled thresholds** instead of arbitrary cutoffs
- **Multiple testing correction** when evaluating many observations
- **Calibrated uncertainty** for downstream decision-making

## Quick Links

- [Installation](installation.md) – Get started in minutes
- [Quick Start](quickstart.md) – Learn the basics with working examples
- [User Guide](user_guide/index.md) – Comprehensive documentation
- [Examples](examples/index.md) – Practical tutorials
- [API Reference](api/index.md) – Complete API documentation

## Key Features

- **Conformal Inference**: Distribution-free uncertainty quantification with finite-sample guarantees
- **Detector Agnostic**: Works with PyOD, scikit-learn, or any custom detector
- **Multiple Strategies**: Split, CrossValidation, Jackknife+, and Bootstrap methods
- **FDR Control**: Control false discovery rates when testing many observations
- **Weighted Conformal**: Handle distribution shift between training and test data

## Installation

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

## Next Steps

Ready to get started? Head to the [Quick Start guide](quickstart.md) to see nonconform in action.
