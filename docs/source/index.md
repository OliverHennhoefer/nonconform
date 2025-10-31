# Welcome to Nonconform's Documentation!

**Nonconform** is a Python library for conformal anomaly detection that provides uncertainty quantification for PyOD detectors. It implements statistically rigorous anomaly detection with p-values and error control metrics like False Discovery Rate (FDR).

## Quick Links

- [Installation](installation.md) - Get started with Nonconform
- [Quick Start](quickstart.md) - Basic usage examples
- [User Guide](user_guide/index.md) - Comprehensive documentation
- [Examples](examples/index.md) - Practical examples and tutorials
- [API Reference](api/index.md) - Complete API documentation
- [Contributing](contributing.md) - How to contribute to the project

## Key Features

- **Conformal Inference**: Distribution-free uncertainty quantification
- **PyOD Integration**: Works with popular anomaly detection library
- **Multiple Strategies**: Split, Bootstrap, Cross-validation, Jackknife+
- **FDR Control**: False Discovery Rate control for multiple testing
- **Weighted Conformal**: Handle non-exchangeable data

## Getting Started

Install Nonconform with pip:

```bash
pip install nonconform
```

Basic usage:

```python
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from pyod.models import IForest

# Create conformal detector
detector = ConformalDetector(IForest(), Split())

# Fit and predict with p-values
detector.fit(X_train)
p_values = detector.predict(X_test)
```
