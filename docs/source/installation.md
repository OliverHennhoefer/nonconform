---
description: "Install nonconform with pip or uv and choose optional extras for PyOD, datasets, online FDR, and probabilistic estimation."
---

# Installation

## Requirements

- Python 3.12 or newer
- A supported platform for NumPy, SciPy, pandas, and scikit-learn

The core package includes batch conformal detection, FDR selection, sequential
conformal monitoring, and conformal martingales.

## Core installation

Use the core installation with supported scikit-learn estimators or a custom
detector.

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

## Optional extras

Install only the capabilities your application needs.

| Extra | Adds | Needed for |
|---|---|---|
| `[pyod]` | [PyOD](https://pyod.readthedocs.io/) | PyOD's detector collection |
| `[data]` | [oddball](https://github.com/OliverHennhoefer/oddball) and PyArrow | Packaged benchmark-dataset workflows |
| `[fdr]` | [online-fdr](https://github.com/OliverHennhoefer/online-fdr) | Online multiple-testing procedures such as GAI and LORD |
| `[probabilistic]` | [KDEpy](https://kdepy.readthedocs.io/) and [Optuna](https://optuna.org/) | `Probabilistic()` KDE estimation and tuning |
| `[all]` | Every optional dependency above | Development or environments that need every feature |

For PyOD models and the example datasets used throughout this site:

=== "pip"
    ```bash
    pip install "nonconform[pyod,data]"
    ```

=== "uv"
    ```bash
    uv add "nonconform[pyod,data]"
    ```

For every optional capability:

=== "pip"
    ```bash
    pip install "nonconform[all]"
    ```

=== "uv"
    ```bash
    uv add "nonconform[all]"
    ```

!!! note "Sequential monitoring needs no extra"

    `nonconform.monitoring` and `nonconform.martingales` belong to the core
    installation. The `[fdr]` extra is for controlling false discoveries across
    hypotheses tested online. It is not required for conformal martingale change
    monitoring, and the two guarantee types are not interchangeable.

## Verify the installation

```python
import nonconform
from nonconform.martingales import SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

print(f"nonconform {nonconform.__version__}")
print(SimpleJumperMartingale.__name__)
print(ExchangeabilityMonitor.__name__)
```

If an optional import fails, verify that its matching extra was installed into
the same Python environment that runs your code.

## Next steps

- Run both core workflows in the [Quick Start](quickstart.md).
- Check supported score interfaces in [Detector Compatibility](user_guide/detector_compatibility.md).
- Review the [API Stability Contract](api/stability.md) before building a reusable integration.
