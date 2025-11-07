# False Discovery Rate Control

This guide explains how to use False Discovery Rate (FDR) control in `nonconform` for multiple testing scenarios. For exchangeable data you can rely on ``scipy.stats.false_discovery_control``. When you work with weighted conformal p-values under covariate shift, use ``nonconform.utils.stat.weighted_false_discovery_control`` to keep statistical guarantees.

## Overview

FDR control is a statistical method for handling multiple hypothesis testing. In anomaly detection, it helps control the proportion of false positives among all detected anomalies. Instead of using a fixed significance level α for all tests, FDR control adjusts the threshold to maintain a desired false discovery rate.

## Basic Usage

```python
import numpy as np
from scipy.stats import false_discovery_control
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation
from pyod.models.lof import LOF

# Prepare detector and data
detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(n_calib=0.2),
    aggregation=Aggregation.MEDIAN,
    seed=42,
)

detector.fit(X_train)
p_values = detector.predict(X_test, raw=False)

# Apply Benjamini–Hochberg control at 5%
adjusted_p_values = false_discovery_control(p_values, method="bh", alpha=0.05)
discoveries = adjusted_p_values < 0.05

print(f"Original detections: {(p_values < 0.05).sum()}")
print(f"FDR-controlled discoveries: {discoveries.sum()}")
```

## Weighted Conformal Selection

When calibration and test data follow different covariate distributions, accompany weighted conformal p-values with Weighted Conformal Selection (WCS). This routine automatically handles the additional randomness introduced by importance weights and pruning.

```python
import numpy as np
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import JackknifeBootstrap
from nonconform.utils.stat import weighted_false_discovery_control
from pyod.models.iforest import IForest

detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=JackknifeBootstrap(n_bootstraps=50),
    weight_estimator=LogisticWeightEstimator(seed=1),
    seed=1,
)

detector.fit(X_train)

# Weighted conformal selection reads all cached quantities from detector.last_result
detector.predict(X_test, raw=False)

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

The ``pruning`` parameter controls the tie-breaking strategy: ``DETERMINISTIC`` uses a fixed rule, while ``HOMOGENEOUS`` and ``HETEROGENEOUS`` introduce shared or independent randomness. Set ``seed`` for reproducible pruning decisions.

## Available Methods

The `scipy.stats.false_discovery_control` function supports several methods:

### Benjamini-Hochberg (BH)
- **Method**: `'bh'`
- **Description**: Most commonly used FDR control method
- **Assumptions**: Independent tests, or tests satisfying positive regression dependence on subsets (PRDS). Note that PRDS is more restrictive than general positive dependence - it requires that for any subset of hypotheses, the joint distribution of p-values is positively dependent.
- **Usage**: `false_discovery_control(p_values, method='bh')`

### Benjamini-Yekutieli (BY)
- **Method**: `'by'`
- **Description**: More conservative method for arbitrary dependence
- **Assumptions**: Works under any dependency structure
- **Usage**: `false_discovery_control(p_values, method='by')`

```python
# Compare different methods
bh_adjusted = false_discovery_control(p_values, method='bh', alpha=0.05)
by_adjusted = false_discovery_control(p_values, method='by', alpha=0.05)

bh_discoveries = (bh_adjusted < 0.05).sum()
by_discoveries = (by_adjusted < 0.05).sum()

print(f"BH discoveries: {bh_discoveries}")
print(f"BY discoveries: {by_discoveries}")
```

## Setting FDR Levels

You can control the desired FDR level using the `alpha` parameter:

```python
# Different FDR levels
fdr_levels = [0.01, 0.05, 0.1, 0.2]

for alpha in fdr_levels:
    adjusted_p_vals = false_discovery_control(p_values, method='bh', alpha=alpha)
    discoveries = (adjusted_p_vals < alpha).sum()
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
1. **Controlled False Discovery Rate**: Maintains the expected proportion of false positives
2. **Increased Power**: Often more powerful than family-wise error rate (FWER) control
3. **Scalability**: Works well with large numbers of tests

### Practical Examples

#### High-dimensional Anomaly Detection
```python
# When analyzing many features independently
n_features = X.shape[1]
feature_p_values = []

for i in range(n_features):
    # Analyze each feature separately
    X_feature = X[:, [i]]
    detector.fit(X_feature)
    p_vals = detector.predict(X_feature, raw=False)
    feature_p_values.extend(p_vals)

# Apply FDR control across all features
all_adjusted = false_discovery_control(feature_p_values, method='bh', alpha=0.05)
```

#### Multiple Time Series
```python
# When analyzing multiple time series
time_series_data = [ts1, ts2, ts3, ...]  # Multiple time series
all_p_values = []

for ts in time_series_data:
    detector.fit(ts)
    p_vals = detector.predict(ts, raw=False)
    all_p_values.extend(p_vals)

# Control FDR across all time series
adjusted_p_vals = false_discovery_control(all_p_values, method='bh', alpha=0.05)
```

## Integration with Conformal Prediction

FDR control works naturally with conformal prediction p-values:

```python
from scipy.stats import false_discovery_control
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.utils.stat import weighted_false_discovery_control
from nonconform.utils.func import Aggregation
from pyod.models.lof import LOF

# Standard conformal detector: use scipy for FDR control
base_detector = LOF()
strategy = Split(n_calib=0.2)

standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42,
)
standard_detector.fit(X_train)
standard_mask = false_discovery_control(
    standard_detector.predict(X_test, raw=False),
    method="bh",
    alpha=0.05,
) < 0.05

# Weighted conformal detector: use Weighted Conformal Selection
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42,
)
weighted_detector.fit(X_train)

weighted_detector.predict(X_test, raw=False)
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

Evaluate the effectiveness of FDR control:

```python
def evaluate_fdr_control(p_values, true_labels, alpha=0.05):
    """Evaluate FDR control performance."""
    # Apply FDR control
    adjusted_p_vals = false_discovery_control(p_values, method='bh', alpha=alpha)
    discoveries = adjusted_p_vals < alpha

    # Calculate metrics
    true_positives = np.sum(discoveries & (true_labels == 1))
    false_positives = np.sum(discoveries & (true_labels == 0))

    if discoveries.sum() > 0:
        empirical_fdr = false_positives / discoveries.sum()
        precision = true_positives / discoveries.sum()
    else:
        empirical_fdr = 0
        precision = 0

    recall = true_positives / np.sum(true_labels == 1) if np.sum(true_labels == 1) > 0 else 0

    return {
        'discoveries': discoveries.sum(),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'empirical_fdr': empirical_fdr,
        'precision': precision,
        'recall': recall
    }

# Example usage
results = evaluate_fdr_control(p_values, y_true, alpha=0.05)
print(f"Empirical FDR: {results['empirical_fdr']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
```

## Best Practices

### 1. Choose Appropriate FDR Level
- **Conservative**: α = 0.01 for critical applications
- **Standard**: α = 0.05 for most applications
- **Liberal**: α = 0.1 when false positives are less costly

### 2. Method Selection
- Use **BH** for most applications (independent or positively dependent tests)
- Use **BY** when tests may have negative dependence or when more conservative control is needed

### 3. Combine with Domain Knowledge
```python
# Incorporate prior knowledge about anomaly prevalence
expected_anomaly_rate = 0.02  # 2% expected anomalies
adjusted_alpha = min(0.05, expected_anomaly_rate * 2)  # Adjust FDR level

adjusted_p_vals = false_discovery_control(p_values, method='bh', alpha=adjusted_alpha)
```

### 4. Monitor Performance
```python
# Track FDR control performance over time
fdr_history = []
for batch in data_batches:
    p_vals = detector.predict(batch, raw=False)
    adj_p_vals = false_discovery_control(p_vals, method='bh', alpha=0.05)
    discoveries = adj_p_vals < 0.05

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
from scipy.stats import combine_pvalues

# Get p-values from multiple detectors
detectors = [LOF(), KNN(), OCSVM()]
p_values_list = []

for detector in detectors:
    conf_detector = ConformalDetector(
        detector=detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    conf_detector.fit(X_train)
    p_vals = conf_detector.predict(X_test, raw=False)
    p_values_list.append(p_vals)

# Combine p-values using Fisher's method
combined_stats, combined_p_values = combine_pvalues(
    np.array(p_values_list).T,
    method='fisher'
)

# Apply FDR control to combined p-values
final_adjusted = false_discovery_control(combined_p_values, method='bh', alpha=0.05)
final_discoveries = final_adjusted < 0.05
```

## Online FDR Control for Streaming Data

For dynamic settings with streaming data batches, the optional `online-fdr` package provides methods that adapt to temporal dependencies while maintaining FDR control.

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
        p_values = detector.predict(batch, raw=False)

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
