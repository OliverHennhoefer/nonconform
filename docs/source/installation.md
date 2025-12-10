# Installation

## Prerequisites

- Python 3.12 or higher

## Basic Installation

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

This installs nonconform with its core dependencies (NumPy, SciPy, scikit-learn). You can use any scikit-learn compatible anomaly detector out of the box.

## Optional Dependencies

nonconform offers optional extras for specific use cases:

| Extra | What it adds | Install when you need |
|-------|-------------|----------------------|
| `[pyod]` | [PyOD](https://pyod.readthedocs.io/) library | Access to 40+ anomaly detection algorithms (Isolation Forest, LOF, KNN, etc.) |
| `[data]` | [oddball](https://github.com/OliverHennhoefer/oddball) + PyArrow | Benchmark datasets for experimentation and testing |
| `[fdr]` | [online-fdr](https://github.com/OliverHennhoefer/online-fdr) | Streaming/online FDR control for real-time applications |
| `[all]` | All of the above | Full functionality |

### Installing Extras

=== "pip"
    ```bash
    # Most common: PyOD detectors + benchmark datasets
    pip install "nonconform[pyod,data]"

    # Full installation with all extras
    pip install "nonconform[all]"
    ```

=== "uv"
    ```bash
    # Most common: PyOD detectors + benchmark datasets
    uv add "nonconform[pyod,data]"

    # Full installation with all extras
    uv add "nonconform[all]"
    ```

### Which Extras Do You Need?

**For most users**, we recommend:

```bash
pip install "nonconform[pyod,data]"
```

This gives you:

- **PyOD**: A large library of anomaly detection algorithms. Most examples in the documentation use PyOD detectors.
- **Data**: Benchmark datasets for experimentation. Useful for learning and testing before applying to your own data.

**Add `[fdr]`** if you need:

- Real-time anomaly detection with streaming FDR control
- Sequential testing over time

## Verify Installation

```python
import nonconform
print(nonconform.__version__)
```

## Next Steps

Head to the [Quick Start](quickstart.md) to see nonconform in action.
