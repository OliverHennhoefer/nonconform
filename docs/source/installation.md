# Installation

Install nonconform with pip or uv.

## Prerequisites

- Python 3.12 or higher

## Installation

=== "pip"
    ```bash
    pip install nonconform
    ```

=== "uv"
    ```bash
    uv add nonconform
    ```

## Optional Dependencies

Available extras: `[pyod]`, `[data]`, `[fdr]`, `[all]`.

- `[pyod]`: PyOD detector support
- `[data]`: Benchmark datasets via `oddball` + `pyarrow`
- `[fdr]`: Online False Discovery Rate control (`online-fdr`)
- `[all]`: All optional extras (`nonconform[pyod,data,fdr]`)

### Installing with Specific Dependencies

=== "pip"
    ```bash
    # With PyOD support
    pip install "nonconform[pyod]"

    # With datasets support
    pip install "nonconform[data]"

    # With online FDR control
    pip install "nonconform[fdr]"

    # All optional dependencies
    pip install "nonconform[all]"
    ```

=== "uv"
    ```bash
    # With PyOD support
    uv add "nonconform[pyod]"

    # With datasets support
    uv add "nonconform[data]"

    # With online FDR control
    uv add "nonconform[fdr]"

    # All optional dependencies
    uv add "nonconform[all]"
    ```

!!! note "Dataset Caching"
    Datasets download automatically on first use and cache both in memory and on disk (`~/.cache/nonconform/`) for faster subsequent access.

## Get Started

You're ready to detect anomalies.

```python
import nonconform
```