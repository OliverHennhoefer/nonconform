# False Discovery Rate Control

## What is FDR and Why Does It Matter?

When you test many observations for anomalies, some will look anomalous by
chance even if they are truly normal. For example, testing 1,000 observations
at a 5% level yields about 50 false positives on average.

**False Discovery Rate (FDR)** is the proportion of false positives among all the observations you flag as anomalies:

$$\text{FDR} = \frac{\text{False Positives}}{\text{Total Discoveries}}$$

**FDR control** adjusts your threshold so that this proportion stays below a
target level (for example, 5%). This differs from controlling false positives
per individual test: FDR controls the error proportion among the points you
actually flag.

!!! example "Example"
    You flag 100 observations with FDR controlled at 5%. On average, no more
    than about 5 of those are false positives. Without FDR control, you might
    flag 200 observations with 50 false positives (25% FDR).

---

## Quick Start

For standard conformal p-values (exchangeable data), use
`conformalized_selection`:

```python
from nonconform.fdr import conformalized_selection

decisions = conformalized_selection(result=detector.last_result, alpha=0.05)
```

For weighted conformal p-values (covariate shift), use `weighted_false_discovery_control`:

!!! note
    `detector.last_result` is populated by the most recent `detector.compute_p_values(...)` call.
    See [Weighted Conformal Selection](#weighted-conformal-selection) below for a complete runnable example.

```python
# After creating and fitting a weighted detector:
# detector = ConformalDetector(...)
# detector.fit(X_train)
# detector.compute_p_values(X_test)

from nonconform.fdr import weighted_false_discovery_control
from nonconform.enums import Pruning

decisions = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42
)
```

---

## Selection Entry Points

Use these functions depending on your setting:

- Standard (exchangeable, unweighted conformal p-values):
- `conformalized_selection(result=...)`
- `conformalized_selection_from_arrays(...)`

- Weighted (covariate shift with importance weights):
- `weighted_false_discovery_control(result=...)`
- `weighted_false_discovery_control_from_arrays(...)`

## Parameter Roles (`delta` vs `alpha`)

When using `ConditionalEmpirical`, keep these roles separate:

- `delta`: calibration confidence/failure budget inside the conditional
  p-value map.
- `alpha`: target FDR level in the final selection rule.

They do not need to be equal. A common pattern is to tune `delta` for p-value
calibration behavior and `alpha` for operational false discovery tolerance.

## Guarantee Scope for `conformalized_selection`

`conformalized_selection` is an explicit BH-style selection step applied to the
provided p-values. Its FDR guarantee depends on:

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
from nonconform.fdr import (
    conformalized_selection,
    conformalized_selection_from_arrays,
    weighted_false_discovery_control,
    weighted_false_discovery_control_from_arrays,
)

# Conformalized Selection (CS, also called conformal BH/cfBH) from result bundle
cs_mask = conformalized_selection(
    result=result,
    alpha=0.05,
)

# CS/cfBH from explicit p-values
cs_from_arrays = conformalized_selection_from_arrays(
    p_values=result.p_values,
    alpha=0.05,
)

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
from nonconform.fdr import conformalized_selection

from pyod.models.lof import LOF

# Prepare detector and data
detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(n_calib=0.2),
    aggregation="median",
    seed=42,
)

detector.fit(X_train)
_ = detector.compute_p_values(X_test)

# Apply conformalized selection (BH-style) at 5%
discoveries = conformalized_selection(
    result=detector.last_result,
    alpha=0.05,
)

print(f"FDR-controlled discoveries: {discoveries.sum()}")
```

## Weighted Conformal Selection

When calibration and test distributions differ, use Weighted Conformal Selection (WCS) with weighted conformal p-values. WCS handles randomness from importance weights and pruning.

```python
import numpy as np
from nonconform import ConformalDetector, JackknifeBootstrap, logistic_weight_estimator
from nonconform.enums import Pruning
from nonconform.fdr import weighted_false_discovery_control
from pyod.models.iforest import IForest

detector = ConformalDetector(
    detector=IForest(random_state=1),
    strategy=JackknifeBootstrap(n_bootstraps=50),
    weight_estimator=logistic_weight_estimator(),
    seed=1,
)

detector.fit(X_train)

# Weighted conformal selection reads all cached quantities from detector.last_result
detector.compute_p_values(X_test)

selected = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=1,
)

print(f"Selected points: {selected.sum()} / {len(selected)}")
```

`ConformalDetector.last_result` always reflects the most recent prediction call,
bundling p-values, scores, and importance weights for downstream analysis.

The ``pruning`` parameter controls tie handling. ``DETERMINISTIC`` uses a fixed
rule. ``HOMOGENEOUS`` and ``HETEROGENEOUS`` use shared or independent
randomness. Set ``seed`` for reproducible randomized pruning decisions.

## Available Methods

`conformalized_selection` uses BH-style selection by default.
If you need direct adjusted p-values or BY control, use
`scipy.stats.false_discovery_control`:

### Benjamini-Hochberg (BH)
- **Method**: `'bh'`
- **Description**: Most commonly used FDR control method
- **Assumptions**: Independent tests, or tests satisfying positive regression
  dependence on subsets (PRDS). In plain terms, PRDS means small p-values tend
  to occur together in a positively dependent way; it is stricter than generic
  "positive dependence."
- **Usage**: `false_discovery_control(p_values, method='bh')`

### Benjamini-Yekutieli (BY)
- **Method**: `'by'`
- **Description**: More conservative method for arbitrary dependence
- **Assumptions**: Works under any dependency structure
- **Usage**: `false_discovery_control(p_values, method='by')`

```python
from scipy.stats import false_discovery_control

# Compare different methods
bh_adjusted = false_discovery_control(p_values, method='bh')
by_adjusted = false_discovery_control(p_values, method='by')

bh_discoveries = (bh_adjusted < 0.05).sum()
by_discoveries = (by_adjusted < 0.05).sum()

print(f"BH discoveries: {bh_discoveries}")
print(f"BY discoveries: {by_discoveries}")
```

## Setting FDR Levels

You can control the desired FDR level using the `alpha` parameter:

```python
from nonconform.fdr import conformalized_selection_from_arrays

# Different FDR levels
fdr_levels = [0.01, 0.05, 0.1, 0.2]

for alpha in fdr_levels:
    discoveries = conformalized_selection_from_arrays(
        p_values=p_values,
        alpha=alpha,
    ).sum()
    print(f"FDR level {alpha}: {discoveries} discoveries")
```

## When to Use FDR Control

### Multiple Testing Scenarios
Use FDR control when:
- Testing multiple hypotheses simultaneously
- Analyzing high-dimensional data
- Processing multiple datasets or time series
- Running ensemble methods with multiple detectors

### Benefits
1. **Controlled FDR**: Bounds expected false positive proportion
2. **Increased Power**: Often more powerful than family-wise error rate (FWER) control
3. **Scalability**: Works well with large numbers of tests

### Practical Examples

#### High-dimensional Anomaly Detection
```python
from nonconform.fdr import conformalized_selection_from_arrays

# When analyzing many features independently
# Use disjoint train/test samples for valid conformal p-values
X_train, X_test = ...  # same feature space, different samples
n_features = X_train.shape[1]
feature_p_values = []

for i in range(n_features):
    # Analyze each feature separately
    X_train_feature = X_train[:, [i]]
    X_test_feature = X_test[:, [i]]
    detector.fit(X_train_feature)
    p_vals = detector.compute_p_values(X_test_feature)
    feature_p_values.extend(p_vals)

# Apply FDR control across all features
discoveries = conformalized_selection_from_arrays(
    p_values=feature_p_values,
    alpha=0.05,
)
```

#### Multiple Time Series
```python
from nonconform.fdr import conformalized_selection_from_arrays

# When analyzing multiple time series
train_series = [ts1_train, ts2_train, ts3_train, ...]
test_series = [ts1_test, ts2_test, ts3_test, ...]
all_p_values = []

for ts_train, ts_test in zip(train_series, test_series):
    detector.fit(ts_train)
    p_vals = detector.compute_p_values(ts_test)
    all_p_values.extend(p_vals)

# Control FDR across all time series
discoveries = conformalized_selection_from_arrays(
    p_values=all_p_values,
    alpha=0.05,
)
```

## Integration with Conformal Prediction

FDR control works naturally with conformal prediction p-values:

```python
from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.enums import Pruning
from nonconform.fdr import (
    conformalized_selection,
    weighted_false_discovery_control,
)
from pyod.models.lof import LOF

# Standard conformal detector: use conformalized selection
base_detector = LOF()
strategy = Split(n_calib=0.2)

standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    seed=42,
)
standard_detector.fit(X_train)
standard_detector.compute_p_values(X_test)
standard_mask = conformalized_selection(
    result=standard_detector.last_result,
    alpha=0.05,
)

# Weighted conformal detector: use Weighted Conformal Selection
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
weighted_detector.fit(X_train)

weighted_detector.compute_p_values(X_test)
weighted_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
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
from nonconform.fdr import conformalized_selection_from_arrays
from nonconform.metrics import false_discovery_rate, statistical_power

def evaluate_fdr_control(p_values, true_labels, alpha=0.05):
    """Evaluate FDR control performance."""
    # Apply FDR control
    discoveries = conformalized_selection_from_arrays(
        p_values=p_values,
        alpha=alpha,
    )

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
- **Conservative**: α = 0.01 for critical applications
- **Standard**: α = 0.05 for most applications
- **Liberal**: α = 0.1 when false positives are less costly

### 2. Method Selection
- Use **`conformalized_selection` (BH-style)** for most conformal workflows
- Use **BY** via SciPy when tests may have negative dependence or when more conservative control is needed

### 3. Combine with Domain Knowledge
```python
from nonconform.fdr import conformalized_selection_from_arrays

# Incorporate prior knowledge about anomaly prevalence
expected_anomaly_rate = 0.02  # 2% expected anomalies
adjusted_alpha = min(0.05, expected_anomaly_rate * 2)  # Adjust FDR level

discoveries = conformalized_selection_from_arrays(
    p_values=p_values,
    alpha=adjusted_alpha,
)
```

### 4. Monitor Performance
```python
from nonconform.fdr import conformalized_selection_from_arrays

# Track FDR control performance over time
fdr_history = []
for batch in data_batches:
    p_vals = detector.compute_p_values(batch)
    discoveries = conformalized_selection_from_arrays(
        p_values=p_vals,
        alpha=0.05,
    )

    if len(true_labels_batch) > 0:  # If ground truth available
        metrics = evaluate_fdr_control(p_vals, true_labels_batch)
        fdr_history.append(metrics['empirical_fdr'])
```

## Common Pitfalls

### 1. Inappropriate Independence Assumptions
- BH assumes independence or positive dependence
- Use BY if negative dependence is suspected

### 2. Multiple Rounds of Testing
- Don't apply FDR control multiple times to the same data
- If doing sequential testing, use specialized methods

### 3. Ignoring Effect Sizes
- FDR control doesn't consider magnitude of anomalies
- Consider combining with effect size thresholds

## Advanced Usage

### Combining Multiple Detection Methods
```python
import numpy as np
from scipy.stats import combine_pvalues
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from nonconform import ConformalDetector, Split
from nonconform.fdr import conformalized_selection_from_arrays


# Get p-values from multiple detectors
detectors = [LOF(), KNN(), OCSVM()]
p_values_list = []
strategy = Split(n_calib=0.2)

for base_detector in detectors:
    conf_detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation="median",
        seed=42
    )
    conf_detector.fit(X_train)
    p_vals = conf_detector.compute_p_values(X_test)
    p_values_list.append(p_vals)

# Combine p-values using Fisher's method
combined_stats, combined_p_values = combine_pvalues(
    np.array(p_values_list).T,
    method='fisher'
)

# Apply FDR control to combined p-values
final_discoveries = conformalized_selection_from_arrays(
    p_values=combined_p_values,
    alpha=0.05,
)
```

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

from onlinefdr import Alpha_investing, LORD

# Example with streaming conformal p-values
def streaming_anomaly_detection(data_stream, detector, alpha=0.05):
    """Online FDR control for streaming anomaly detection."""

    # Initialize online FDR method
    # Alpha-investing: adapts alpha based on discoveries
    online_fdr = Alpha_investing(alpha=alpha, w0=0.05)

    discoveries = []

    for batch in data_stream:
        # Get p-values for current batch
        p_values = detector.compute_p_values(batch)

        # Apply online FDR control
        for p_val in p_values:
            decision = online_fdr.run_single(p_val)
            discoveries.append(decision)

    return discoveries
```

### LORD (Levels based On Recent Discovery) Method

```python
# LORD method: more aggressive when recent discoveries
lord_fdr = LORD(alpha=0.05, tau=0.5)

# Process streaming data with temporal adaptation
for t, (batch, p_values) in enumerate(stream_with_pvalues):
    for p_val in p_values:
        # LORD adapts rejection threshold based on recent discoveries
        reject = lord_fdr.run_single(p_val)

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
