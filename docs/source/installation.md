# Installation

## Prerequisites

- Python 3.12 or higher

## Installation Profiles

Pick an installation profile based on how you want to get started.

### 1. Core (minimal dependencies)

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

This includes NumPy, SciPy, and scikit-learn.

### 2. Anomaly-ready (recommended for most users)

=== "pip"
    ```bash
    pip install "nonconform[pyod,data]"
    ```

=== "uv"
    ```bash
    uv add "nonconform[pyod,data]"
    ```

This adds:
- PyOD detector zoo (`[pyod]`)
- oddball benchmark datasets (`[data]`)

### 3. Full installation

=== "pip"
    ```bash
    pip install "nonconform[all]"
    ```

=== "uv"
    ```bash
    uv add "nonconform[all]"
    ```

## Optional Dependencies

nonconform offers optional extras for specific use cases:

| Extra | What it adds | Install when you need |
|-------|-------------|----------------------|
| `[pyod]` | [PyOD](https://pyod.readthedocs.io/) library | Access to 40+ anomaly detection algorithms (Isolation Forest, LOF, KNN, etc.) |
| `[data]` | [oddball](https://github.com/OliverHennhoefer/oddball) + PyArrow | Benchmark datasets for experimentation and testing |
| `[fdr]` | [online-fdr](https://github.com/OliverHennhoefer/online-fdr) | Streaming/online FDR control for real-time applications |
| `[all]` | All of the above | Full functionality |

### Which Extras Do You Need?

- Add `[pyod]` if you want a larger set of anomaly detectors.
- Add `[data]` if you want oddball benchmark datasets.
- Add `[fdr]` if you need:

- Real-time anomaly detection with streaming FDR control
- Sequential testing over time

## Verify Installation

```python
import nonconform
print(nonconform.__version__)
```

## Next Steps

Head to the [Quick Start](quickstart.md) to see nonconform in action.
