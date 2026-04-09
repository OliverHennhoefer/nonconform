# Integrative Conformal

This example demonstrates the labeled-inlier/labeled-outlier workflow with
`IntegrativeConformalDetector`.

## Setup

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from nonconform import (
    IntegrativeConformalDetector,
    IntegrativeModel,
    IntegrativeSplit,
)

rng = np.random.default_rng(42)

X_inliers = rng.normal(loc=0.0, scale=0.7, size=(120, 3))
X_outliers = rng.normal(loc=3.0, scale=0.7, size=(80, 3))

X_test = np.vstack(
    [
        rng.normal(loc=0.1, scale=0.8, size=(20, 3)),
        rng.normal(loc=3.1, scale=0.8, size=(20, 3)),
    ]
)
```

## Configure Models

```python
from sklearn.ensemble import IsolationForest

models = [
    IntegrativeModel.one_class(
        reference="inlier",
        estimator=IsolationForest(random_state=42),
        score_polarity="auto",
    ),
    IntegrativeModel.one_class(
        reference="outlier",
        estimator=IsolationForest(random_state=42),
        score_polarity="auto",
    ),
    IntegrativeModel.binary(
        estimator=LogisticRegression(),
        inlier_label=0,
    ),
]
```

## Fit And Predict

```python
detector = IntegrativeConformalDetector(
    models=models,
    strategy=IntegrativeSplit(n_calib=0.2),
    seed=42,
)

detector.fit(X_inliers, X_outliers)

p_values = detector.compute_p_values(X_test)
scores = detector.score_samples(X_test)
selected = detector.select(X_test, alpha=0.1)

print(f"P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
print(f"Ratio score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"Discoveries: {selected.sum()}")
```

## TCV+ Variant

```python
from nonconform import TransductiveCVPlus

tcv_detector = IntegrativeConformalDetector(
    models=models,
    strategy=TransductiveCVPlus(k_in=5, k_out=5),
    seed=42,
)

tcv_detector.fit(X_inliers, X_outliers)
tcv_p_values = tcv_detector.compute_p_values(X_test)
print(f"TCV+ min p-value: {tcv_p_values.min():.4f}")
```

`TransductiveCVPlus` currently supports p-value computation but not the paper's
dedicated TCV+ conditional-FDR selection routine.
