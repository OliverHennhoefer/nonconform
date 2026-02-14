# Logging and Progress Control

`nonconform` uses Python logging for strategy-level progress and warnings.
Aggregation progress bars are controlled by `ConformalDetector(verbose=...)`.

## Quick Setup

```python
import logging

# Development: show progress/info
logging.getLogger("nonconform").setLevel(logging.INFO)

# Production: warnings and errors only
logging.getLogger("nonconform").setLevel(logging.WARNING)
```

## `verbose` vs Logging Level

- `verbose=True` shows aggregation progress during `compute_p_values()` and
  `score_samples()`.
- Logging level controls strategy-level progress output and warnings.

```python
import logging
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, CrossValidation

logging.getLogger("nonconform").setLevel(logging.INFO)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=CrossValidation(k=5),
    score_polarity="auto",
    verbose=True,
    seed=42,
)
detector.fit(X_train)
_ = detector.compute_p_values(X_test)
```

## Common Patterns

### Quiet Production Output

```python
import logging

logging.getLogger("nonconform").setLevel(logging.WARNING)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    verbose=False,
)
```

### Keep Aggregation Bars, Hide Strategy Details

```python
import logging

logging.getLogger("nonconform").setLevel(logging.WARNING)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    verbose=True,
)
```

### Full Debugging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

## Typical Progress Messages

| Message | Component | Trigger |
|---|---|---|
| `CV fold training (N folds)` | Cross-validation strategy | `fit()` |
| `Bootstrap training (N folds)` | Bootstrap strategy | `fit()` |
| `Aggregating N models` | Ensemble aggregation | `compute_p_values()` / `score_samples()` with `verbose=True` |

## Logger Names

Useful logger namespaces:

- `nonconform`
- `nonconform.resampling.*`
- `nonconform.weighting.*`
- `nonconform.fdr`
- `nonconform.adapters`
- `nonconform._internal.*`

Configure specific namespaces if you want different levels per subsystem.

```python
import logging

# Base level for package logs
logging.getLogger("nonconform").setLevel(logging.INFO)

# More detail for resampling strategies only
logging.getLogger("nonconform.resampling").setLevel(logging.DEBUG)
```
