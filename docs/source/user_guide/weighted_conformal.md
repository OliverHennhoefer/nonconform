# Weighted Conformal P-values

This guide explains how to use weighted conformal p-values in `nonconform` for handling distribution shift and covariate shift scenarios.

## Overview

Weighted conformal p-values extend classical conformal prediction to handle covariate shift scenarios [[Jin & Candès, 2023](#references); [Tibshirani et al., 2019](#references)]. **Key assumption**: the marginal distribution P(X) may change between calibration and test data, while the conditional distribution P(Y|X) – the relationship between features and anomaly status – remains constant. This assumption is crucial for the validity of weighted conformal inference. When it holds you can pair the p-values with Weighted Conformal Selection (WCS) to obtain rigorous False Discovery Rate control under distribution shift [[Jin & Candès, 2023](#references)].

The `ConformalDetector` with a `weight_estimator` parameter automatically estimates importance weights to distinguish between calibration and test samples, then uses these weights to compute adjusted p-values.

## Basic Usage

```python
import numpy as np
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.utils.func.enums import Aggregation
from pyod.models.lof import LOF

# Initialize base detector
base_detector = LOF()
strategy = Split(n_calib=0.2)

# Create weighted conformal detector
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)

# Fit on training data
detector.fit(X_train)

# Get weighted p-values for test data
# The detector automatically computes importance weights
p_values = detector.predict(X_test, raw=False)
```

## How It Works

The weighted conformal method works through the following steps:

### 1. Calibration
During fitting, the detector:
- Uses the specified strategy to split data and train models
- Computes calibration scores on held-out calibration data
- Stores calibration samples for later weight computation

### 2. Weight Estimation
During prediction, the detector:
- Trains a logistic regression model to distinguish calibration from test samples
- Uses the predicted probabilities to estimate importance weights
- Applies weights to both calibration and test instances

### 3. Weighted P-value Calculation
The p-values are computed using weighted empirical distribution functions:

```python
# Simplified version of the weighted p-value calculation
def weighted_p_value(test_score, calibration_scores, calibration_weights, test_weight):
    """
    Calculate weighted conformal p-value with proper tie handling.

    The p-value represents the probability of observing a score
    at least as extreme as the test score under the weighted
    calibration distribution.
    """
    # Count calibration scores strictly greater than test score
    weighted_rank = np.sum(calibration_weights[calibration_scores > test_score])

    # Handle ties: add random fraction of tied weights (coin flip approach)
    tied_weights = np.sum(calibration_weights[calibration_scores == test_score])
    weighted_rank += np.random.uniform(0, 1) * tied_weights

    # Add test instance weight (always included for conformal guarantee)
    weighted_rank += test_weight
    total_weight = np.sum(calibration_weights) + test_weight

    return weighted_rank / total_weight
```

## When to Use Weighted Conformal

### Covariate Shift Scenarios
Use weighted conformal detection when:

1. **Domain Adaptation**: Training on one domain, testing on another
2. **Temporal Shift**: Data distribution changes over time
3. **Sample Selection Bias**: Test data is not representative of training data
4. **Stratified Sampling**: Different sampling rates for different subgroups

### Examples of Distribution Shift

```python
# Example 1: Temporal shift
# Training data from 2020, test data from 2024
detector.fit(X_train_2020)
p_values_2024 = detector.predict(X_test_2024, raw=False)

# Example 2: Geographic shift
# Training on US data, testing on European data
detector.fit(X_us)
p_values_europe = detector.predict(X_europe, raw=False)

# Example 3: Sensor drift
# Calibration data before sensor drift, test data after
detector.fit(X_before_drift)
p_values_after_drift = detector.predict(X_after_drift, raw=False)
```

## Comparison with Standard Conformal

```python
# Standard conformal detector
standard_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Weighted conformal detector
weighted_detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)

# Fit both on training data
standard_detector.fit(X_train)
weighted_detector.fit(X_train)

# Compare on shifted test data
standard_p_values = standard_detector.predict(X_test_shifted, raw=False)
weighted_p_values = weighted_detector.predict(X_test_shifted, raw=False)

# Apply FDR control for proper comparison
from scipy.stats import false_discovery_control
from nonconform.utils.stat import weighted_false_discovery_control

standard_mask = false_discovery_control(standard_p_values, method="bh") < 0.05

weighted_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"Standard conformal detections: {standard_mask.sum()}")
print(f"Weighted conformal detections: {weighted_mask.sum()}")
```

## Different Aggregation Strategies

The choice of aggregation method can affect performance under distribution shift:

```python
# Compare different aggregation methods
aggregation_methods = [Aggregation.MEAN, Aggregation.MEDIAN, Aggregation.MAX]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=agg_method,
        weight_estimator=LogisticWeightEstimator(seed=42),
        seed=42
    )
    detector.fit(X_train)
    p_vals = detector.predict(X_test_shifted, raw=False)

    print(f"{agg_method.value}: {(p_vals < 0.05).sum()} detections")
```

## Weight Estimators

`nonconform` provides two weight estimators for handling covariate shift:

### LogisticWeightEstimator

Uses logistic regression to estimate likelihood ratios between calibration and test distributions:

```python
from nonconform.detection.weight import LogisticWeightEstimator

# Default configuration with L2 regularization
weight_est = LogisticWeightEstimator(seed=42)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=weight_est,
    seed=42
)
```

**When to use**:
- Linear or moderately complex distribution shifts
- High-dimensional data where interpretability matters
- Fast weight estimation is needed
- Default choice for most applications

**Parameters**:
- `penalty`: Regularization type ('l2', 'l1', 'elasticnet', 'none')
- `C`: Inverse regularization strength (default: 1.0)
- `solver`: Optimization algorithm (default: 'lbfgs')
- `max_iter`: Maximum iterations (default: 1000)
- `seed`: Random state for reproducibility

### ForestWeightEstimator

Uses random forest classification to estimate likelihood ratios:

```python
from nonconform.detection.weight import ForestWeightEstimator

# Forest-based weight estimation
weight_est = ForestWeightEstimator(
    n_estimators=100,
    max_depth=10,
    seed=42
)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=weight_est,
    seed=42
)
```

**When to use**:
- Complex, non-linear distribution shifts
- Feature interactions are important
- More robust to outliers in feature space
- When you have sufficient calibration data (hundreds+ samples)

**Parameters**:
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)
- `min_samples_split`: Minimum samples to split (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)
- `seed`: Random state for reproducibility

### Comparison

```python
# Compare weight estimators on complex shift
from nonconform.detection.weight import LogisticWeightEstimator, ForestWeightEstimator

estimators = {
    'Logistic': LogisticWeightEstimator(seed=42),
    'Forest': ForestWeightEstimator(n_estimators=100, seed=42)
}

for name, weight_est in estimators.items():
    detector = ConformalDetector(
        detector=base_detector,
        strategy=Split(n_calib=0.2),
        aggregation=Aggregation.MEDIAN,
        weight_estimator=weight_est,
        seed=42
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test_shifted, raw=False)

    print(f"{name}: {(p_values < 0.05).sum()} detections")
```

**General recommendations**:
- Start with `LogisticWeightEstimator` (faster, more interpretable)
- Switch to `ForestWeightEstimator` if:
  - Distribution shift is highly non-linear
  - You have >500 calibration samples
  - Logistic weights show poor discrimination

### BootstrapBaggedWeightEstimator

Wraps any base weight estimator with bootstrap bagging for improved stability in extreme imbalance scenarios:

```python
from nonconform.detection.weight.wrapper.bagging import BootstrapBaggedWeightEstimator
from nonconform.detection.weight import ForestWeightEstimator

# Bootstrap bagging with forest base (best for extreme imbalance)
weight_est = BootstrapBaggedWeightEstimator(
    base_estimator=ForestWeightEstimator(n_estimators=50, seed=42),
    n_bootstrap=50,
    clip_bounds=(0.35, 45.0),
    clip_quantile=0.05
)

detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=1000),
    aggregation=Aggregation.MEDIAN,
    weight_estimator=weight_est,
    seed=42
)
```

#### How It Works

Bootstrap bagging creates an ensemble of weight estimators:

1. **For each bootstrap iteration** (n_bootstrap times):
   - Resample both calibration and test sets to balanced size
   - Fit the base estimator on the bootstrap sample
   - Score ALL original instances (perfect coverage)
   - Store log-weights for each instance

2. **After all iterations**:
   - Aggregate using geometric mean (exp of mean log-weights)
   - Apply clipping to maintain bounded weights

Every instance receives exactly n_bootstrap weight estimates, ensuring symmetric coverage regardless of set size ratios.

#### When to Use

**DO use BootstrapBaggedWeightEstimator when:**

1. **Extreme imbalance**: Large calibration set (>1000) with small test batches (<50)
   - Common in online/streaming detection
   - Example: 1000 calibration samples, 25 test instances

2. **High-stakes applications**: Where weight quality is critical
   - Medical diagnosis with small patient batches
   - Fraud detection with limited transactions
   - Safety-critical systems

3. **Severe distribution shift**: When base estimators produce extreme weights

**DO NOT use for:**

1. **Balanced or moderate imbalance**: Marginal benefit (2-3% improvement) doesn't justify 2-5x computational overhead
2. **Large test sets**: Benefits diminish with larger batches
3. **Latency-sensitive production**: Significant computational cost (20-50x slower)

#### Performance Benchmarks

Empirical testing shows context-dependent value:

##### Balanced Scenario (1000 calib vs 1000 test)
| Metric | Base | Bagged-50 | Improvement |
|--------|------|-----------|-------------|
| Weight Std | 2.884 | 2.957 | **-2.5%** (worse) |
| Extreme Weights | 0 | 0 | No change |
| Time | 0.14s | 0.34s | 2.4x slower |

**Verdict**: Not recommended for balanced sets.

##### Extreme Imbalance (1000 calib vs 25 test)
| Metric | Logistic Base | Logistic Bagged-50 | Improvement |
|--------|---------------|-------------------|-------------|
| Weight Std | 1.604 | 0.841 | **48% better** |
| Extreme Weights | 612 | 385 | 37% reduction |
| Recall | 0.067 | 0.200 | **3x better** |
| Time | 0.14s | 0.34s | 2.4x slower |

| Metric | Forest Base | Forest Bagged-50 | Improvement |
|--------|-------------|------------------|-------------|
| Weight Std | 0.153 | 0.259 | Slightly higher but stable |
| Extreme Weights | 599 | **0** | **100% elimination** |
| Recall | 0.333 | **1.000** | **Perfect detection** |
| FDR | 0.000 | 0.190 | Acceptable trade-off |
| Time | 0.24s | 6.4s | 27x slower |

**Verdict**: **Strongly recommended** for extreme imbalance. Best combination: `ForestWeightEstimator + Bagging`.

#### Configuration Parameters

**n_bootstrap** (default: 100):
- Number of bootstrap iterations
- Higher = more stable, but slower
- Recommended: 20-50 for small test batches, 50-100 for critical applications

**clip_bounds** (default: (0.35, 45.0)):
- Fixed clipping bounds for weights after aggregation
- Prevents extreme values that could destabilize p-value computation
- Use when you have domain knowledge about reasonable weight ranges

**clip_quantile** (default: 0.05):
- Adaptive quantile-based clipping
- Clips to (quantile, 1-quantile) percentiles
- Use when weight distribution is unknown
- Set to None to use fixed clip_bounds instead

#### Advanced Example: Streaming Detection

For online/streaming anomaly detection with small batches:

```python
from nonconform.detection import ConformalDetector
from nonconform.detection.weight import ForestWeightEstimator
from nonconform.detection.weight.wrapper.bagging import BootstrapBaggedWeightEstimator
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation

# Configuration for small batch streaming
weight_est = BootstrapBaggedWeightEstimator(
    base_estimator=ForestWeightEstimator(
        n_estimators=50,
        max_depth=10,
        seed=42
    ),
    n_bootstrap=50,
    clip_quantile=0.05  # Adaptive clipping
)

detector = ConformalDetector(
    detector=IForest(behaviour="new", random_state=42),
    strategy=Split(n_calib=1000),  # Large calibration set
    aggregation=Aggregation.MEDIAN,
    weight_estimator=weight_est,
    seed=42
)

# Train on historical data
detector.fit(X_historical)

# Process small incoming batches
for X_batch in stream_data(batch_size=25):
    p_values = detector.predict(X_batch, raw=False)

    # Apply weighted FDR control
    from nonconform.utils.stat import weighted_false_discovery_control
    from nonconform.utils.func.enums import Pruning

    discoveries = weighted_false_discovery_control(
        result=detector.last_result,
        alpha=0.1,
        pruning=Pruning.DETERMINISTIC,
        seed=42
    )

    print(f"Detected {discoveries.sum()} anomalies in batch of {len(X_batch)}")
```

#### Cost-Benefit Analysis

| Configuration | Time | Quality | Use Case |
|--------------|------|---------|----------|
| Logistic (Base) | 0.14s | Baseline | Standard balanced scenarios |
| Logistic + Bagging(50) | 0.34s | +48% weight stability | Moderate imbalance, quality focus |
| Forest (Base) | 0.24s | Good for non-linear | Standard scenarios |
| **Forest + Bagging(50)** | **6.4s** | **Perfect detection** | **Extreme imbalance, premium quality** |

**Recommendation**: Use `ForestWeightEstimator + BootstrapBaggedWeightEstimator` when:
- Calibration set is 40x larger than test batch (e.g., 1000:25)
- Missing anomalies is very costly
- Computational budget allows 20-50x overhead
- Online/streaming detection with small batches

### Decision Guide

**Which weight estimator should I use?**

```
┌─ Is your test batch very small (<50) AND calibration large (>1000)?
│
├─ YES → BootstrapBaggedWeightEstimator(
│         ForestWeightEstimator(50), n_bootstrap=50
│       )
│       Cost: High (6-7s), Quality: Best (perfect detection)
│
└─ NO → Standard weight estimators
    │
    ├─ Linear/moderate shift → LogisticWeightEstimator()
    │                          Cost: Low (0.14s), Quality: Good
    │
    └─ Complex/non-linear shift → ForestWeightEstimator(50)
                                   Cost: Medium (0.24s), Quality: Better
```

## Strategy Selection

Different strategies can be used with weighted conformal detection:

```python
from nonconform.strategy import Bootstrap
from nonconform.strategy import CrossValidation

# Bootstrap strategy for stability
bootstrap_strategy = Bootstrap(n_bootstraps=100, resampling_ratio=0.8)
bootstrap_detector = ConformalDetector(
    detector=base_detector,
    strategy=bootstrap_strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)

# Cross-validation strategy for efficiency
cv_strategy = CrossValidation(k=5)
cv_detector = ConformalDetector(
    detector=base_detector,
    strategy=cv_strategy,
    aggregation=Aggregation.MEDIAN,
    weight_estimator=LogisticWeightEstimator(seed=42),
    seed=42
)
```

## Weighted Conformal Selection

Weighted conformal p-values are valid on their own. To obtain finite-sample FDR control under covariate shift, combine them with Weighted Conformal Selection (WCS) [[Jin & Candès, 2023](#references)]:

```python
from nonconform.utils.stat import weighted_false_discovery_control
from nonconform.utils.func.enums import Pruning

# Collect weighted p-values and cached statistics
weighted_detector.predict(X_test_shifted, raw=False)

wcs_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"WCS-selected anomalies: {wcs_mask.sum()} of {len(wcs_mask)}")
```

After any call to `predict()`, the detector caches the relevant arrays `(p_values, scores, weights)` inside `detector.last_result`. Passing this object to `weighted_false_discovery_control` avoids plumbing the raw arrays manually.

### Pruning Modes

The `pruning` parameter controls how ties and randomization are handled in the WCS procedure [[Jin & Candès, 2023](#references)]:

#### Pruning.DETERMINISTIC

```python
wcs_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,  # seed has no effect for deterministic mode
)
```

**Behavior**: Uses a deterministic threshold without randomization. When there are tied p-values at the threshold, includes all or none based on deterministic rule.

**When to use**:
- Reproducibility is critical
- You don't want any randomness in selections
- Reporting results that must be exactly reproducible

**Trade-off**: May be slightly conservative (reject fewer hypotheses) compared to randomized methods.

#### Pruning.HOMOGENEOUS

```python
wcs_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.HOMOGENEOUS,
    seed=42,  # controls randomization
)
```

**Behavior**: Draws a single uniform random variable and applies the same randomized threshold to all test instances. Handles ties by probabilistically including tied instances.

**When to use**:
- Default randomized method
- Want exact FDR control in expectation
- Acceptable to have some randomness

**Trade-off**: Less conservative than DETERMINISTIC, but results vary across random seeds.

#### Pruning.HETEROGENEOUS

```python
wcs_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.HETEROGENEOUS,
    seed=42,  # controls randomization
)
```

**Behavior**: Draws independent uniform random variables for each test instance. Provides the most flexible randomization.

**When to use**:
- Maximum power (fewer false negatives)
- Most aggressive FDR control
- Research settings where slight variance is acceptable

**Trade-off**: Highest variance across random seeds, but best expected power.

### Comparison of Pruning Methods

```python
from nonconform.utils.func.enums import Pruning

pruning_methods = [
    Pruning.DETERMINISTIC,
    Pruning.HOMOGENEOUS,
    Pruning.HETEROGENEOUS
]

weighted_detector.predict(X_test_shifted, raw=False)

for pruning_method in pruning_methods:
    wcs_mask = weighted_false_discovery_control(
        result=weighted_detector.last_result,
        alpha=0.05,
        pruning=pruning_method,
        seed=42,
    )

    print(f"{pruning_method.name}: {wcs_mask.sum()} detections")
```

**Expected relationship**: Typically HETEROGENEOUS ≥ HOMOGENEOUS ≥ DETERMINISTIC in terms of number of detections, though this can vary with data.


## Performance Considerations

### Computational Cost
Weighted conformal detection has additional overhead:
- Weight estimation via logistic regression
- Weighted p-value computation

```python
import time

# Compare computation times
def time_detector(detector, X_train, X_test):
    start_time = time.time()
    detector.fit(X_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    p_values = detector.predict(X_test, raw=False)
    predict_time = time.time() - start_time

    return fit_time, predict_time

# Standard vs Weighted timing
standard_fit, standard_pred = time_detector(standard_detector, X_train, X_test)
weighted_fit, weighted_pred = time_detector(weighted_detector, X_train, X_test)

print(f"Standard: Fit={standard_fit:.2f}s, Predict={standard_pred:.2f}s")
print(f"Weighted: Fit={weighted_fit:.2f}s, Predict={weighted_pred:.2f}s")
print(f"Overhead: {((weighted_fit + weighted_pred) / (standard_fit + standard_pred) - 1) * 100:.1f}%")
```

### Memory Usage
Weighted conformal detection requires storing:
- Calibration samples for weight computation
- Calibration scores for p-value calculation

For large datasets, consider:
- Using a subset of calibration samples for weight estimation
- Implementing online/streaming versions

## Best Practices

### 1. Validate Distribution Shift
Always check if distribution shift is actually present:

```python
# Use statistical tests to detect shift
from scipy.stats import ks_2samp

def detect_feature_shift(X_train, X_test):
    """Detect distribution shift in individual features."""
    shift_detected = []
    p_values = []

    for i in range(X_train.shape[1]):
        statistic, p_value = ks_2samp(X_train[:, i], X_test[:, i])
        shift_detected.append(p_value < 0.05)
        p_values.append(p_value)

    print(f"Features with significant shift: {sum(shift_detected)}/{len(shift_detected)}")
    return shift_detected, p_values

shift_features, shift_p_values = detect_feature_shift(X_train, X_test_shifted)
```

### 2. Combine with Weighted Conformal Selection

```python
from nonconform.utils.stat import weighted_false_discovery_control

weighted_p_values = weighted_detector.predict(X_test_shifted, raw=False)
wcs_mask = weighted_false_discovery_control(
    result=weighted_detector.last_result,
    alpha=0.05,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"Raw detections: {(weighted_p_values < 0.05).sum()}")
print(f"WCS-controlled detections: {wcs_mask.sum()}")
```

### 3. Monitor Weight Quality
Extreme weights can indicate poor weight estimation:

```python
def check_weight_quality(detector, X_calib, X_test):
    """Check for extreme weights that might indicate poor estimation."""
    # This is a conceptual example - actual implementation would require
    # access to the internal weights computed by the detector

    # Rule of thumb: weights should typically be between 0.1 and 10
    # Extreme weights (< 0.01 or > 100) suggest problems
    pass
```

### 4. Use Appropriate Base Detectors
Some detectors work better with weighted conformal:
- **Good**: Distance-based methods (LOF, KNN) that are sensitive to distribution
- **Moderate**: Tree-based methods (Isolation Forest) that are somewhat robust
- **Challenging**: Neural networks that might already adapt to shift

## Advanced Applications

### Multi-domain Adaptation
```python
# Handle multiple domains with different shift patterns
domains = ['domain_A', 'domain_B', 'domain_C']
domain_detectors = {}

for domain in domains:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        weight_estimator=LogisticWeightEstimator(seed=42),
        seed=42
    )
    detector.fit(X_train)  # Common training set
    domain_detectors[domain] = detector

# Predict on domain-specific test sets
for domain in domains:
    X_test_domain = load_domain_data(domain)  # Load domain-specific test data
    p_values = domain_detectors[domain].predict(X_test_domain, raw=False)
    print(f"{domain}: {(p_values < 0.05).sum()} detections")
```

### Online Adaptation
```python
# Adapt to gradual distribution shift over time
def online_weighted_detection(detector, data_stream, window_size=1000):
    """Online weighted conformal detection with sliding window."""
    detections = []

    for i, (X_batch, _) in enumerate(data_stream):
        if i == 0:
            # Initialize with first batch
            detector.fit(X_batch)
        else:
            # Use sliding window for calibration
            if i * len(X_batch) > window_size:
                start_idx = (i * len(X_batch)) - window_size
                X_calib = get_recent_data(start_idx, window_size)
                detector.fit(X_calib)

            # Predict on current batch
            p_values = detector.predict(X_batch, raw=False)
            batch_detections = (p_values < 0.05).sum()
            detections.append(batch_detections)

    return detections
```

## Troubleshooting

### Common Issues

1. **Poor Weight Estimation**
   - Insufficient calibration data
   - High-dimensional data with small samples
   - Solution: Increase calibration size or use dimensionality reduction

2. **Extreme P-values**
   - All p-values near 0 or 1
   - Solution: Check for severe distribution shift or model mismatch

3. **Inconsistent Results**
   - High variance in detection counts
   - Solution: Use bootstrap strategy or increase sample size

### Debugging Tools
```python
def debug_weighted_conformal(detector, X_train, X_test):
    """Debug weighted conformal detection issues."""
    print("=== Weighted Conformal Debug Report ===")

    # Check data properties
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")

    # Fit detector
    detector.fit(X_train)

    # Check calibration set size
    print(f"Calibration samples: {len(detector.calibration_set)}")

    if len(detector.calibration_set) < 50:
        print("WARNING: Small calibration set may lead to unreliable weights")

    # Get predictions
    p_values = detector.predict(X_test, raw=False)

    # Check p-value distribution
    print(f"P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
    print(f"P-value mean: {p_values.mean():.4f}")
    print(f"P-value std: {p_values.std():.4f}")

    if p_values.std() < 0.01:
        print("WARNING: Very low p-value variance - check for issues")

    print("=== End Debug Report ===")

# Example usage
debug_weighted_conformal(weighted_detector, X_train, X_test_shifted)
```

## References

- **Jin, Y., & Candès, E. J. (2023)**. *Model-free Selective Inference Under Covariate Shift via Weighted Conformal p-values*. Biometrika, 110(4), 1090-1106. arXiv:2307.09291. [Foundational paper on weighted conformal inference and WCS procedure]

- **Tibshirani, R. J., Barber, R. F., Candes, E., & Ramdas, A. (2019)**. *Conformal Prediction Under Covariate Shift*. Advances in Neural Information Processing Systems, 32. arXiv:1904.06019. [Early work on conformal prediction with covariate shift]

- **Genovese, C. R., Roeder, K., & Wasserman, L. (2006)**. *False Discovery Control with p-value Weighting*. Biometrika, 93(3), 509-524. [Theoretical foundation for weighted FDR control]

## Next Steps

- Learn about [FDR control](fdr_control.md) for multiple testing scenarios
- Explore [different conformalization strategies](conformalization_strategies.md) for various use cases
- Read about [best practices](best_practices.md) for robust anomaly detection
- Check the [troubleshooting guide](troubleshooting.md) for common issues
- See [input validation](input_validation.md) for parameter constraints and edge cases
