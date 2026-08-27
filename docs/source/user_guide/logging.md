---
description: "Configure nonconform logger levels and progress bars for fitting, aggregation, weighting, and weighted FDR control."
---

# Logging and progress

`nonconform` uses standard Python logger levels to decide whether several
long-running operations display `tqdm` progress bars. Raw-score aggregation has
a separate `ConformalDetector(verbose=...)` switch.

## Controls at a glance

| Output | Control | Display label |
|---|---|---|
| Cross-validation fitting | `nonconform.resampling.crossval` enabled for `INFO` | `Calibration` |
| Jackknife-bootstrap fitting | `nonconform.resampling.bootstrap` enabled for `INFO` | `Calibration` |
| Bootstrap-bagged weight fitting | `nonconform.weighting.bagged` enabled for `INFO` | `Weighting` |
| WCS iteration | `nonconform.fdr` enabled for `INFO` | `Weighted FDR Control` |
| Aggregating retained detector scores | `ConformalDetector(verbose=True)` | `Aggregation` |

Warnings use the same logger hierarchy. `verbose=False` does not suppress
warnings, and changing the logger level does not disable an aggregation bar
explicitly requested with `verbose=True`.

## Set the package level explicitly

Configure logging before constructing and fitting detectors:

```python
import logging

package_logger = logging.getLogger("nonconform")
package_logger.setLevel(logging.WARNING)

print(logging.getLevelName(package_logger.level))
```

Useful package-wide levels are:

- `logging.DEBUG` for parameter-normalization details;
- `logging.INFO` for strategy, weighting, and WCS progress;
- `logging.WARNING` for warnings and errors only; and
- `logging.ERROR` for errors only.

The effective behavior can also depend on handlers configured by the host
application. Libraries should not call `logging.basicConfig(...)` on behalf of
an application.

## Complete progress example

```python
import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, CrossValidation

logging.getLogger("nonconform").setLevel(logging.INFO)

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(150, 3))
x_test = rng.normal(size=(20, 3))

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=20, random_state=42),
    strategy=CrossValidation(k=3),
    verbose=True,
    seed=42,
).fit(x_reference)

p_values = detector.compute_p_values(x_test)
print(p_values[:3])
```

This can show `Calibration` during the three fold fits and `Aggregation` while
the retained models score `x_test`.

## Configure one subsystem

Logger names follow Python's dotted hierarchy:

```python
import logging

logging.getLogger("nonconform").setLevel(logging.WARNING)
logging.getLogger("nonconform.resampling.bootstrap").setLevel(logging.INFO)

print(
    logging.getLevelName(
        logging.getLogger("nonconform.resampling.bootstrap").getEffectiveLevel()
    )
)
```

The relevant names are:

- `nonconform`
- `nonconform.adapters`
- `nonconform.resampling.crossval`
- `nonconform.resampling.bootstrap`
- `nonconform.weighting.bagged`
- `nonconform.fdr`

Set a child logger explicitly only when its output policy should differ from
the package-level policy.

## Application logging pattern

For scripts, configure a handler once at the application boundary, then set the
package level:

```python
import logging

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(levelname)s %(name)s: %(message)s")
)

logger = logging.getLogger("nonconform")
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

logger.info("nonconform logging configured")
```

Clearing handlers is appropriate in a standalone script that owns its logging
configuration. Do not do it inside reusable library code or a hosted runtime
whose handlers belong to the caller.

Progress bars normally write to standard error. Account for that when capturing
logs in notebooks, tests, job runners, or container platforms.
