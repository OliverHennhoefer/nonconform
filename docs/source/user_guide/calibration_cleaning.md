---
description: "Use Label-Trim helpers to clean contaminated calibration scores with a limited annotation budget."
---

# Calibration Cleaning

Conformal outlier detection is most direct when the calibration reference data
contains verified normal samples. In practice, reference data can be
contaminated by a small number of hidden outliers. `nonconform.cleaning`
provides score-level Label-Trim helpers for this setting.

The implementation follows the Label-Trim workflow from Bashari, Sesia, and
Romano, *Robust Conformal Outlier Detection under Contaminated Reference Data*
(ICML 2025). It is an opt-in preprocessing step: existing p-value, FDR,
weighting, and detector behavior are unchanged.

## Score Workflow

Use this workflow when you already have calibration and test anomaly scores.
Scores must be anomaly-oriented: higher means more anomalous.

```python
import numpy as np

from nonconform.cleaning import apply_label_trim, select_label_trim_candidates
from nonconform.scoring import Empirical

# Use any already-fitted anomaly detector or score function.
calibration_scores = np.asarray(base_detector.decision_function(X_calib))
test_scores = np.asarray(base_detector.decision_function(X_test))

candidate_indices = select_label_trim_candidates(
    calibration_scores,
    label_budget=50,
)

# Labels come from an expert or trusted audit of only these candidates.
# Convention: 0 = normal, 1 = outlier.
candidate_labels = [...]

trim = apply_label_trim(
    calibration_scores,
    candidate_indices,
    candidate_labels,
)

p_values = Empirical().compute_p_values(test_scores, trim.trimmed_scores)
```

`apply_label_trim(...)` removes only annotated candidate points whose label is
the configured outlier label. Unannotated calibration points and annotated
normal points remain in the calibration set.

## Detached Calibration Workflow

Use the returned `keep_mask` when you want to calibrate a pre-fitted detector on
cleaned calibration samples.

```python
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.cleaning import apply_label_trim, select_label_trim_candidates

base_detector = IsolationForest(random_state=42)
base_detector.fit(X_fit)

raw_scores = base_detector.decision_function(X_calib)
candidate_indices = select_label_trim_candidates(raw_scores, label_budget=50)
candidate_labels = [...]

trim = apply_label_trim(raw_scores, candidate_indices, candidate_labels)

detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.2),
    score_polarity="auto",
)
detector.calibrate(X_calib[trim.keep_mask])
discoveries = detector.select(X_test, alpha=0.05)
```

## Validity Notes

- Label-Trim depends on trusted labels for the selected high-score candidates.
- Do not relabel the same data after looking at test outcomes.
- This helper does not add a weighted-conformal-specific guarantee. If you use
  weighted workflows, treat cleaning as a preprocessing decision and keep the
  usual weighted conformal assumptions in force.
- Keep the label budget and candidate-selection rule fixed before evaluating
  downstream FDR or power.

## References

- Bashari, Sesia, and Romano (2025),
  [Robust Conformal Outlier Detection under Contaminated Reference Data](https://openreview.net/forum?id=s55Af9Emyq).
- Official implementation:
  [Meshiba/robust-conformal-od](https://github.com/Meshiba/robust-conformal-od).
