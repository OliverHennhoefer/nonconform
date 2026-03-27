# False Discovery Rate Control

## What is FDR and Why Does It Matter?

When you test many observations for anomalies, some will look anomalous by
chance even if they are truly normal. For example, testing 1,000 observations
at significance level alpha = 0.05 yields about 50 false positives on average.

**False Discovery Rate (FDR)** is the proportion of false positives among all the observations you flag as anomalies:

$$\text{FDR} = \frac{\text{False Positives}}{\text{Total Discoveries}}$$

An equivalent operational interpretation is:

$$\text{FDR} \approx \frac{\text{Wasted Effort (chasing false positives)}}{\text{Total Investigation Effort}}$$

**FDR control** adjusts your threshold so that this proportion stays below a
target level (for example, 5%). This differs from controlling false positives
per individual test: FDR controls the error proportion among the points you
actually flag.

!!! example "Example"
    Suppose your pipeline flags 100 observations as anomalies with
    `alpha = 0.05` FDR control.

    - Expected false alarms: about 5
    - Useful follow-ups: about 95

    Now compare this to an uncontrolled setup that flags 200 observations,
    where 50 are false positives:

    - False positives: 50/200 = 25% FDR
    - This means 1 in 4 investigations is wasted effort

---

## Quick Start

`detector.select()` is the recommended single-call entry point. It combines
p-value computation with the appropriate FDR-controlled selection procedure,
automatically dispatching to weighted selection when a `weight_estimator` is
configured:

```python
detector.fit(X_train)
mask = detector.select(X_test, alpha=0.05)
```

For the weighted case with custom pruning:

```python
from nonconform.enums import Pruning

mask = detector.select(
    X_test,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)
```

When you need raw p-values for custom downstream analysis (multi-alpha sweeps,
combining detectors, etc.), use `compute_p_values(...)` plus SciPy BH:

```python
from scipy.stats import false_discovery_control

p_values = detector.compute_p_values(X_test)
decisions = false_discovery_control(p_values, method="bh") <= 0.05
```

!!! note
    `detector.last_result` is populated by the most recent
    `detector.compute_p_values(...)` or `detector.select(...)` call.
    See [Weighted Conformal Selection](#weighted-conformal-selection) below for
    a complete runnable example.

---

## Selection Entry Points

**Primary (recommended):** `detector.select(X_test, alpha=...)` â€” dispatches
automatically based on detector configuration; no manual result-bundle
handling required.

**Advanced/low-level options** (for custom workflows):

- Standard (exchangeable): apply BH directly via
  `scipy.stats.false_discovery_control(...)` to conformal p-values.
- Weighted (covariate shift with importance weights):
  `weighted_false_discovery_control(result=...)` or
  `weighted_false_discovery_control_from_arrays(...)`.

## Parameter Roles (`delta` vs `alpha`)

When using `ConditionalEmpirical`, keep these roles separate:

- `delta`: calibration confidence/failure budget inside the conditional
  p-value map.
- `alpha`: target FDR level in the final selection rule.

They do not need to be equal. A common pattern is to tune `delta` for p-value
calibration behavior and `alpha` for operational false discovery tolerance.

## Guarantee Scope for BH-Style Selection

BH-style selection applied to conformal p-values has guarantees that depend on:

- how valid/calibrated those p-values are,
- exchangeability (or the relevant data-shift assumptions for weighted methods),
- and BH dependence assumptions (independence or PRDS).

In other words, the selection routine itself does not create validity from
invalid inputs; it preserves guarantees under the assumptions above.

!!! warning "Strict validation for weighted inputs"
    Weighted FDR routines fail fast on invalid inputs.
    They now raise `ValueError` when:

    - score/weight arrays are not 1D numeric arrays of matching lengths
    - any score/weight/p-value contains non-finite values
    - any weight is negative
    - total calibration weight is not strictly positive
    - `result.metadata["kde"]` is present but malformed
      (missing keys, invalid shapes, non-monotone grid/CDF, or non-positive total weight)

```python
from scipy.stats import false_discovery_control
from nonconform.fdr import (
    weighted_false_discovery_control,
    weighted_false_discovery_control_from_arrays,
)

# Standard BH selection from explicit p-values
cs_mask = false_discovery_control(result.p_values, method="bh") <= 0.05

# Strict WCS from cached result bundle
wcs_from_result = weighted_false_discovery_control(
    result=result,
    alpha=0.05,
)

# Strict WCS from explicit arrays
wcs_mask = weighted_false_discovery_control_from_arrays(
    p_values=result.p_values,
    test_scores=result.test_scores,
    calib_scores=result.calib_scores,
    test_weights=result.test_weights,
    calib_weights=result.calib_weights,
    alpha=0.05,
)
```

---

## Basic Usage

```python
from nonconform import ConformalDetector, Split

from pyod.models.lof import LOF

detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(n_calib=0.2),
    aggregation="median",
    seed=42,
)

detector.fit(X_train)

# FDR-controlled selection at 5% â€” single call
discoveries = detector.select(X_test, alpha=0.05)

print(f"FDR-controlled discoveries: {discoveries.sum()}")
```

## Weighted Conformal Selection

When calibration and test distributions differ, configure a `weight_estimator`
and call `select()` â€” it automatically dispatches to Weighted Conformalized
Selection (WCS):

```python
from nonconform import ConformalDetector, JackknifeBootstrap, logistic_weight_estimator
from nonconform.enums import Pruning
from pyod.models.iforest import IForest

detector = ConformalDetector(
    detector=IForest(random_state=1),
    strategy=JackknifeBootstrap(n_bootstraps=50),
    weight_estimator=logistic_weight_estimator(),
    seed=1,
)

detector.fit(X_train)

selected = detector.select(
    X_test,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=1,
)

print(f"Selected points: {selected.sum()} / {len(selected)}")
```

The ``pruning`` parameter controls tie handling. ``DETERMINISTIC`` uses a fixed
rule. ``HOMOGENEOUS`` and ``HETEROGENEOUS`` use shared or independent
randomness. Set ``seed`` for reproducible randomized pruning decisions.

## Available Methods

For direct BH control on conformal p-values, use
`scipy.stats.false_discovery_control`:

### Benjamini-Hochberg (BH)
- **Method**: `'bh'`
- **Description**: Most commonly used FDR control method
- **Assumptions**: Independent tests, or tests satisfying positive regression
  dependence on subsets (PRDS). In plain terms, PRDS means small p-values tend
  to occur together in a positively dependent way; it is stricter than generic
  "positive dependence."
- **Usage**: `false_discovery_control(p_values, method='bh')`

```python
from scipy.stats import false_discovery_control

# BH control on conformal p-values
bh_adjusted = false_discovery_control(p_values, method='bh')
bh_discoveries = (bh_adjusted < 0.05).sum()

print(f"BH discoveries: {bh_discoveries}")
```

## Setting FDR Levels

You can control the desired FDR level using the `alpha` parameter:

```python
from scipy.stats import false_discovery_control

# Different FDR levels
fdr_levels = [0.01, 0.05, 0.1, 0.2]

for alpha in fdr_levels:
    discoveries = (false_discovery_control(p_values, method="bh") <= alpha).sum()
    print(f"FDR level {alpha}: {discoveries} discoveries")
```

## When to Use FDR Control

Use FDR control whenever you make more than one test-level anomaly decision.
This includes both batch decisions made simultaneously and decisions accumulated
over time.

### Core Rule
- One test: a per-test threshold may be enough.
- Multiple tests: control FDR to bound the expected fraction of false
  discoveries among flagged points.

### Why
1. **Controlled false discoveries**: bounds expected false-positive proportion among detections.
2. **Practical power trade-off**: usually more powerful than stricter family-wise error control.
3. **Scales to many tests**: suitable for modern high-throughput anomaly workflows.

### Sequential Note
If decisions are made over time (not a fixed batch), use procedures designed
for online settings (see [Online FDR Control for Streaming Data](#online-fdr-control-for-streaming-data)).

## Integration with Conformal Prediction

`select()` dispatches automatically â€” standard or weighted â€” based on the
detector's configuration:

```python
from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.enums import Pruning
from pyod.models.lof import LOF

base_detector = LOF()
strategy = Split(n_calib=0.2)

# Standard: BH-style FDR selection on conformal p-values
standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    seed=42,
)
standard_detector.fit(X_train)
standard_mask = standard_detector.select(X_test, alpha=0.05)

# Weighted: WCS (handles covariate shift via importance weights)
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
weighted_detector.fit(X_train)
weighted_mask = weighted_detector.select(
    X_test,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"Standard detections: {standard_mask.sum()}")
print(f"Weighted detections: {weighted_mask.sum()}")
```

## Performance Evaluation

Evaluate the effectiveness of FDR control using nonconform's built-in metrics:

```python
from scipy.stats import false_discovery_control
from nonconform.metrics import false_discovery_rate, statistical_power

def evaluate_fdr_control(p_values, true_labels, alpha=0.05):
    """Evaluate FDR control performance."""
    # Apply FDR control
    discoveries = false_discovery_control(p_values, method="bh") <= alpha

    # Calculate metrics using nonconform functions
    empirical_fdr = false_discovery_rate(true_labels, discoveries)
    power = statistical_power(true_labels, discoveries)

    return {
        'discoveries': discoveries.sum(),
        'empirical_fdr': empirical_fdr,
        'power': power
    }

# Example usage
results = evaluate_fdr_control(p_values, y_true, alpha=0.05)
print(f"Discoveries: {results['discoveries']}")
print(f"Empirical FDR: {results['empirical_fdr']:.3f}")
print(f"Statistical Power: {results['power']:.3f}")
```

## Best Practices

### 1. Choose Appropriate FDR Level
- **Very strict**: `alpha = 0.01` only when false positives are extremely costly (often too strict for exploratory workflows)
- **Standard**: `alpha = 0.05` for most applications
- **Exploratory / higher-recall**: `alpha = 0.10` when missing anomalies is costlier than investigating additional false positives

### 2. Method Selection
- Use **`detector.select(...)`** for most conformal workflows
- Use **BH** via SciPy for manual p-value thresholding workflows

### 3. Combine with Domain Knowledge
```python
from scipy.stats import false_discovery_control

# Incorporate prior knowledge about anomaly prevalence
expected_anomaly_rate = 0.02  # 2% expected anomalies
adjusted_alpha = min(0.05, expected_anomaly_rate * 2)  # Adjust FDR level

discoveries = false_discovery_control(p_values, method="bh") <= adjusted_alpha
```

### 4. Monitor Performance
```python
from scipy.stats import false_discovery_control

# Track FDR control performance over time
fdr_history = []
for batch in data_batches:
    p_vals = detector.compute_p_values(batch)
    discoveries = false_discovery_control(p_vals, method="bh") <= 0.05

    if len(true_labels_batch) > 0:  # If ground truth available
        metrics = evaluate_fdr_control(p_vals, true_labels_batch)
        fdr_history.append(metrics['empirical_fdr'])
```

## Common Pitfalls

### 1. Inappropriate Independence Assumptions
- BH assumes independence or positive dependence
- Re-check assumptions or move to methods designed for your dependence structure

### 2. Multiple Rounds of Testing
- Don't apply FDR control multiple times to the same data
- If doing sequential testing, use specialized methods

## Online FDR Control for Streaming Data

For dynamic settings with streaming data batches, the optional `online-fdr` package provides methods that adapt to temporal dependencies while maintaining FDR control.

Do not conflate this with martingale alarm thresholds such as
`ville_threshold` in [Exchangeability Martingales](exchangeability_martingales.md):
those provide anytime false-alarm control on evidence processes, not FDR
control across multiple tested hypotheses.

### Installation and Basic Usage

```python
# Install FDR dependencies
# pip install nonconform[fdr]

from online_fdr.investing.alpha.alpha import Gai
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale
from nonconform.temporal import TemporalSession

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(X_train)

# GAI alpha-investing controller
controller = Gai(alpha=0.05, wealth=0.025)
session = TemporalSession(
    detector=detector,
    online_controller=controller,
    martingale=PowerMartingale(epsilon=0.5),
)

for batch in data_stream:
    result = session.step(batch, apply_batch_select=True, alpha=0.05)
    print(result.online_decisions.sum(), result.triggered_alarms)
```

`TemporalSession` is the recommended streaming orchestration path because it
keeps p-value generation, online FDR decisions, and martingale updates in one
stateful interface. The controller itself still follows the online-fdr contract
(`test_one(float) -> bool`).

### LORD (Levels based On Recent Discovery) Method

```python
from online_fdr.investing.lord.three import LordThree

# LORD 3: alpha allocation adapts over the testing stream
lord_fdr = LordThree(alpha=0.05, wealth=0.04, reward=0.05)

# Process streaming data with temporal adaptation
for t, (batch, p_values) in enumerate(stream_with_pvalues):
    for p_val in p_values:
        # LORD adapts rejection threshold based on recent discoveries
        reject = lord_fdr.test_one(float(p_val))

        if reject:
            print(f"Anomaly detected at time {t} with p-value {p_val:.4f}")
```

### Statistical Assumptions for Online FDR

**Key Requirements:**
- **Independence assumption**: Test statistics should be independent or satisfy specific dependency structures
- **Sequential testing**: Methods designed for sequential hypothesis testing scenarios
- **Temporal stability**: Underlying anomaly detection model should be reasonably stable

**When NOT to use online FDR:**
- Strong temporal dependencies in p-values without proper correction
- Concept drift affecting p-value calibration
- Non-stationary data streams requiring model retraining

**Best practice**: Combine with windowed model retraining and exchangeability monitoring for robust streaming anomaly detection.

## Next Steps

- Learn about [weighted conformal p-values](weighted_conformal.md) for handling distribution shift
- Explore [different conformalization strategies](conformalization_strategies.md) for various scenarios
- Read about [best practices](best_practices.md) for robust anomaly detection
