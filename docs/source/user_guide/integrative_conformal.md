# Integrative Conformal

Use integrative conformal detection when you have **labeled inliers and labeled
outliers** and want conformal p-values that combine both sources of evidence.

!!! abstract "When to use this"
    Choose `IntegrativeConformalDetector` when you have:

    - a labeled inlier sample
    - a labeled outlier sample
    - an unlabeled test batch you want to score or select from

    Stay with `ConformalDetector` when you only have inliers plus optional
    covariate-shift weighting.

## Why this is separate from weighted conformal

`nonconform` already supports *weighted conformal* methods for covariate shift.
Those methods correct distribution shift between calibration and test data.

Integrative conformal is different:

- it assumes access to labeled outliers
- it builds two preliminary conformal signals, one from inliers and one from
  outliers
- it recombines them through an additional conformal calibration layer
- it uses its own split conditional-FDR selection procedure

Because the assumptions and downstream selection rule differ materially, the API
is separate instead of overloading `weight_estimator` or `ConditionalEmpirical`.

## Core API

```python
from sklearn.linear_model import LogisticRegression

from nonconform import (
    IntegrativeConformalDetector,
    IntegrativeModel,
    IntegrativeSplit,
)

models = [
    IntegrativeModel.one_class(
        reference="inlier",
        estimator=your_inlier_detector,
        score_polarity="higher_is_anomalous",
    ),
    IntegrativeModel.one_class(
        reference="outlier",
        estimator=your_outlier_detector,
        score_polarity="higher_is_anomalous",
    ),
    IntegrativeModel.binary(
        estimator=LogisticRegression(),
        inlier_label=0,
    ),
]

detector = IntegrativeConformalDetector(
    models=models,
    strategy=IntegrativeSplit(n_calib=0.2),
    seed=42,
)
detector.fit(X_inliers, X_outliers)

p_values = detector.compute_p_values(X_test)
scores = detector.score_samples(X_test)  # r = u0 / u1
selected = detector.select(X_test, alpha=0.1)
```

## Strategies

### IntegrativeSplit

`IntegrativeSplit` implements the paper's split-conformal integrative method.

- splits inliers and outliers independently into training and calibration sets
- computes preliminary `u0` and `u1`
- combines them through `r = u0 / u1`
- re-conformalizes `r` against the inlier calibration set
- supports the paper's split conditional-FDR selector via `select(...)`

### TransductiveCVPlus

`TransductiveCVPlus` implements the TCV+ p-value path.

- uses fold-based held-out scoring instead of a single calibration split
- is usually more expensive than `IntegrativeSplit`
- currently supports `compute_p_values(...)` and `score_samples(...)`
- intentionally does **not** expose `select(...)` yet, because the paper's TCV+
  conditional-FDR selector is substantially more expensive and is not included
  in this first release

## Model specifications

### One-class models

```python
IntegrativeModel.one_class(
    reference="inlier",
    estimator=your_detector,
    score_polarity="higher_is_anomalous",
)
```

`reference="inlier"` means the detector is trained on inliers and contributes to
the `u0` path. `reference="outlier"` means it is trained on outliers and
contributes to the `u1` path.

### Binary models

```python
IntegrativeModel.binary(
    estimator=your_classifier,
    inlier_label=0,
    score_source="auto",
)
```

`score_source` controls how nonconform extracts a continuous score from the
binary estimator:

- `"auto"` (recommended default): use `predict_proba` when available, otherwise
  fall back to `decision_function`
- `"predict_proba"`: force probability-based scoring
- `"decision_function"`: force margin/decision-value scoring

Set an explicit value when your estimator exposes both methods and you want to
lock in one scoring interface for reproducibility, or when `auto` picks a
source you do not want.

Binary scores are standardized internally so higher values mean "more
inlier-like". The detector then applies sign tuning as needed when selecting the
best inlier-side and outlier-side model for each test point.

## Outputs

After `compute_p_values(...)`, `last_result` contains:

- `p_values`: final integrative conformal p-values
- `test_scores`: final ratio statistic `r = u0 / u1`
- `calib_scores`: per-test calibration ratios used in the final re-conformalization
- `metadata["integrative"]`: cached split or TCV+ artifacts, selected model
  indices, sign choices, and strategy-specific debug information

## Selection behavior

For `IntegrativeSplit`, `select(...)` runs the paper's split conditional-FDR
procedure:

```python
mask = detector.select(X_test, alpha=0.1)
```

This is **not** BH, BY, or weighted FDR control. It is a dedicated procedure for
integrative conformal p-values.

For `TransductiveCVPlus`, `select(...)` raises a clear `NotImplementedError`.

## References

- Liang, Z., Sesia, M., & Sun, W. (2024). *Integrative conformal p-values for
  out-of-distribution testing with labelled outliers.* Journal of the Royal
  Statistical Society Series B: Statistical Methodology, 86(3), 671-693.
  https://doi.org/10.1093/jrsssb/qkad138. Preprint: arXiv:2208.11111.
- Repository reference:
  [weighted_conformal_pvalues](https://github.com/ZiyiLiang/weighted_conformal_pvalues)
