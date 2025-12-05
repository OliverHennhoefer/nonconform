# Welcome to Nonconform's Documentation!

**Nonconform** provides uncertainty quantification for anomaly detectors. It wraps PyOD, scikit-learn, or custom detectors to produce statistically valid p-values with False Discovery Rate (FDR) control.

## Quick Links

- [Installation](installation.md) - Get started with Nonconform
- [Quick Start](quickstart.md) - Basic usage examples
- [User Guide](user_guide/index.md) - Comprehensive documentation
- [Examples](examples/index.md) - Practical examples and tutorials
- [API Reference](api/index.md) - Complete API documentation
- [Contributing](contributing.md) - How to contribute to the project

## Key Features

- **Conformal Inference**: Distribution-free uncertainty quantification
- **Detector Agnostic**: Works with PyOD, scikit-learn, or custom detectors via the AnomalyDetector protocol
- **Multiple Strategies**: Split, Bootstrap, Cross-validation, Jackknife+
- **FDR Control**: False Discovery Rate control for multiple testing
- **Weighted Conformal**: Handle non-exchangeable data

## Getting Started

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

Basic usage:

```python
from nonconform import ConformalDetector, Split
from pyod.models.iforest import IForest

# Create conformal detector
detector = ConformalDetector(IForest(), Split())

# Fit and predict with p-values
detector.fit(X_train)
p_values = detector.predict(X_test)
```
