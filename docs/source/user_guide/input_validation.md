# Input Validation and Error Handling

This guide explains parameter constraints, validation rules, and error handling in nonconform. Understanding these constraints will help you avoid common pitfalls and debug issues more effectively.

## Core Parameter Constraints

### Calibration Set Size (`n_calib`)

The calibration set size is the most critical parameter affecting the statistical properties of your conformal predictions.

#### Value Types and Ranges

```python
from nonconform.strategy import Split

# Absolute count (integer)
strategy = Split(n_calib=2000)  # Use exactly 2000 samples for calibration

# Proportional split (float between 0 and 1)
strategy = Split(n_calib=0.2)   # Use 20% of training data for calibration
```

**Valid values**:
- **Float**: `0 < n_calib < 1` - Proportion of training data
- **Integer**: `1 <= n_calib < n_train` - Absolute number of samples

**Constraints**:
- Must be strictly positive
- Cannot exceed the size of the training data
- For proportional splits, must be less than 1.0

#### P-value Resolution

With `n` calibration points, you can obtain at most **n+1 distinct p-values**:

$$\text{Possible p-values} = \left\{\frac{1}{n+1}, \frac{2}{n+1}, \ldots, \frac{n}{n+1}, \frac{n+1}{n+1}\right\}$$

**Example**: With `n_calib=99`, you get 100 possible p-values: {0.01, 0.02, 0.03, ..., 0.99, 1.00}

**Practical Guidelines**:
- **Minimum recommended**: 50-100 samples for reasonable p-value resolution
- **For FDR control at α=0.05**: Consider `n_calib >= 100` for at least 20 distinct p-values below threshold
- **For precise control**: Use `n_calib >= 1000` for fine-grained p-value resolution

#### Validation Example

```python
def validate_calibration_size(X_train, n_calib):
    """Validate calibration set size parameter."""
    n_train = len(X_train)

    # Check type and range
    if isinstance(n_calib, float):
        if not (0 < n_calib < 1):
            raise ValueError(
                f"Proportional n_calib must be in (0, 1), got {n_calib}"
            )
        n_calib_abs = int(n_calib * n_train)
    elif isinstance(n_calib, int):
        if not (1 <= n_calib < n_train):
            raise ValueError(
                f"Absolute n_calib must be in [1, {n_train}), got {n_calib}"
            )
        n_calib_abs = n_calib
    else:
        raise TypeError(
            f"n_calib must be int or float, got {type(n_calib)}"
        )

    # Warn about low resolution
    if n_calib_abs < 50:
        import warnings
        warnings.warn(
            f"Small calibration set (n={n_calib_abs}) will have limited "
            f"p-value resolution ({n_calib_abs + 1} distinct values). "
            f"Consider using at least 50-100 calibration samples."
        )

    return n_calib_abs

# Usage
n_calib_validated = validate_calibration_size(X_train, n_calib=0.2)
```

### Cross-Validation Folds (`k`)

For cross-validation-based strategies:

```python
from nonconform.strategy import CrossValidation

strategy = CrossValidation(k=5)
```

**Valid values**:
- **Integer**: `2 <= k <= n_train`
- Common choices: 5, 10

**Constraints**:
- Must be at least 2 (otherwise no cross-validation)
- Cannot exceed training set size
- Each fold will have approximately `n_train / k` samples

**Trade-offs**:
- **Larger k**: Better statistical efficiency, higher computational cost
- **Smaller k**: Faster computation, potentially less stable estimates

### Bootstrap Parameters (`n_bootstraps`, `ratio`)

For bootstrap-based strategies:

```python
from nonconform.strategy.experimental import Bootstrap

strategy = Bootstrap(n_bootstraps=100, ratio=0.5)
```

**Valid values**:
- `n_bootstraps`: Integer >= 1 (typical: 20-200)
- `ratio`: Float in (0, 1] - proportion of data in each bootstrap sample

**Constraints**:
- More bootstraps improve stability but increase computation
- `ratio` controls calibration set size (lower = more data for training)

### Random Seed (`seed`)

```python
detector = ConformalDetector(detector=base_det, strategy=strategy, seed=42)
```

**Valid values**:
- **Integer**: Any valid Python integer
- **None**: Non-reproducible random behavior

**Purpose**:
- Ensures reproducible train/calibration splits
- Critical for scientific reproducibility

## Edge Cases and Special Behaviors

### Case 1: Very Small Calibration Sets (n_calib ≤ 10)

**What happens**:
```python
# Only 10 calibration samples
strategy = Split(n_calib=10)
detector = ConformalDetector(detector=IsolationForest(), strategy=strategy)
detector.fit(X_train)
p_values = detector.predict(X_test, raw=False)

# p_values will only take 11 distinct values: {1/11, 2/11, ..., 11/11}
# Very coarse resolution!
```

**Consequences**:
- Extremely limited p-value resolution (only 11 distinct values with n=10)
- Large prediction set sizes (conservative predictions)
- Inconsistent performance across different random seeds
- FDR control may be ineffective

**Recommendation**: Avoid calibration sets smaller than 50 samples unless you understand the limitations.

### Case 2: All Calibration Scores Identical

**What happens**:
```python
# Pathological case: all calibration data has same score
# (e.g., constant features, deterministic detector)
```

**Behavior**:
- All test scores ≥ calibration scores → p-value = 1.0 (treated as normal)
- All test scores < calibration scores → p-value = 1/(n+1) (treated as anomalous)
- Binary classification with no gradation

**Detection**:
```python
def check_calibration_diversity(calib_scores, tolerance=1e-10):
    """Check if calibration scores have sufficient diversity."""
    if np.ptp(calib_scores) < tolerance:  # peak-to-peak
        raise ValueError(
            "Calibration scores are nearly identical. "
            "This may indicate: (1) constant features, "
            "(2) detector not properly trained, or "
            "(3) insufficient data diversity."
        )

# Access last calibration scores
calib_scores = detector.last_result.calibration_scores
check_calibration_diversity(calib_scores)
```

### Case 3: Test Score Outside Calibration Range

**What happens**:
```python
# Test score is extremely anomalous (beyond all calibration scores)
test_score = 100.0  # Far beyond max calibration score
# → p-value = 1/(n+1) (smallest possible)

# Test score is extremely normal (below all calibration scores)
test_score = -100.0  # Far below min calibration score
# → p-value = 1.0 (largest possible)
```

**Behavior**:
- Conformal p-values are **bounded**: always in [1/(n+1), 1]
- Extreme test scores saturate at boundary values
- This is correct behavior - conformal prediction provides conservative guarantees

### Case 4: Insufficient Training Data

**What happens**:
```python
# Only 100 training samples, but requesting 200 for calibration
X_train = np.random.randn(100, 10)
strategy = Split(n_calib=200)  # More than available!

detector = ConformalDetector(detector=LOF(), strategy=strategy)
# → Will raise ValueError during fit()
```

**Error message**:
```
ValueError: Calibration size (200) exceeds training size (100)
```

**Solution**: Reduce `n_calib` or provide more training data.

### Case 5: Zero Training Data for Detector

**What happens**:
```python
# Using all data for calibration leaves none for training
X_train = np.random.randn(100, 10)
strategy = Split(n_calib=100)  # All data!

detector = ConformalDetector(detector=LOF(), strategy=strategy)
# → Will raise ValueError
```

**Error message**:
```
ValueError: No training data remaining after calibration split.
Reduce n_calib to leave data for training the base detector.
```

**Solution**: Ensure `n_calib < n_train` to leave data for training.

## Weight Estimator Validation

When using weighted conformal inference:

```python
from nonconform.estimation.weight import LogisticWeightEstimator

weight_est = LogisticWeightEstimator(penalty='l2', C=1.0)
detector = ConformalDetector(
    detector=base_det,
    strategy=strategy,
    weight_estimator=weight_est
)
```

**Constraints**:
- Weight estimator must have `fit()` and `predict_proba()` methods
- Must be compatible with binary classification (calibration vs. test)
- Output probabilities should be in (0, 1)

**Common issues**:
```python
# Issue 1: Weights too extreme (near 0 or infinity)
weights = weight_est.predict_proba(X_test)[:, 1] / (1 - weight_est.predict_proba(X_calib).mean())
if np.any(weights > 1000) or np.any(weights < 0.001):
    import warnings
    warnings.warn("Extreme weight values detected. Covariate shift may be too severe.")

# Issue 2: Perfect separation (weights undefined)
# Occurs when calibration and test are completely separable
# → LogisticRegression may not converge
```

## Aggregation Method Constraints

```python
from nonconform.utils.func import Aggregation

detector = ConformalDetector(
    detector=base_det,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN  # or MEAN, MAX
)
```

**Valid values**:
- `Aggregation.MEAN`: Average p-values across splits (default for many strategies)
- `Aggregation.MEDIAN`: Median p-value (more robust to outliers)
- `Aggregation.MAX`: Most conservative (maximum p-value)

**When it matters**:
- Only relevant for strategies that produce multiple p-values (CrossValidation, Jackknife, Bootstrap)
- Split strategy computes single p-value → aggregation has no effect

## Validation Best Practices

### 1. Always Validate Input Data

```python
def validate_training_data(X_train, y_train=None):
    """Validate training data before fitting."""
    # Check for NaN/Inf
    if not np.all(np.isfinite(X_train)):
        raise ValueError("Training data contains NaN or Inf values")

    # Check dimensions
    if X_train.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X_train.shape}")

    # Check sufficient samples
    if len(X_train) < 100:
        import warnings
        warnings.warn(
            f"Small training set (n={len(X_train)}). "
            f"Consider collecting more data for stable conformal predictions."
        )

    # For one-class anomaly detection, y_train should be all normal (0)
    if y_train is not None and np.any(y_train != 0):
        import warnings
        warnings.warn(
            "Training labels contain anomalies (y != 0). "
            "Conformal anomaly detection assumes one-class training."
        )

validate_training_data(X_train, y_train)
```

### 2. Check Exchangeability Assumptions

```python
def check_exchangeability(X_train, X_test):
    """Simple heuristic check for potential exchangeability violations."""
    from scipy.stats import ks_2samp

    # Compare feature distributions
    violations = []
    for i in range(X_train.shape[1]):
        stat, pval = ks_2samp(X_train[:, i], X_test[:, i])
        if pval < 0.001:  # Strong evidence of distribution shift
            violations.append(i)

    if violations:
        import warnings
        warnings.warn(
            f"Features {violations} show significant distribution shift. "
            f"Consider using weighted conformal inference."
        )

check_exchangeability(X_train, X_test)
```

### 3. Validate Predictions

```python
def validate_p_values(p_values):
    """Validate conformal p-values after prediction."""
    # Check range
    if not np.all((p_values >= 0) & (p_values <= 1)):
        raise ValueError("P-values must be in [0, 1]")

    # Check for invalid values
    if not np.all(np.isfinite(p_values)):
        raise ValueError("P-values contain NaN or Inf")

    # Check for degenerate case (all same value)
    if np.std(p_values) < 1e-10:
        import warnings
        warnings.warn(
            "P-values have near-zero variance. "
            "This may indicate issues with calibration or detector."
        )

p_values = detector.predict(X_test, raw=False)
validate_p_values(p_values)
```

## Error Messages Reference

### Common Errors and Fixes

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: n_calib must be positive` | `n_calib <= 0` | Use positive integer or float in (0, 1) |
| `ValueError: n_calib exceeds training size` | `n_calib >= len(X_train)` | Reduce n_calib or add more training data |
| `ValueError: No training data after split` | All data used for calibration | Ensure n_calib < n_train |
| `TypeError: n_calib must be int or float` | Wrong type passed | Use int (count) or float (proportion) |
| `ValueError: P-values outside [0,1]` | Internal calculation error | Check for data issues or file bug report |
| `ConvergenceWarning` from weight estimator | Calibration and test too different | Covariate shift may be too severe |

## Summary Checklist

Before fitting a conformal detector, verify:

- [ ] Training data has sufficient samples (n ≥ 100 recommended)
- [ ] `n_calib` leaves enough data for training the base detector
- [ ] `n_calib` provides adequate p-value resolution (≥ 50-100 recommended)
- [ ] No NaN or Inf values in training data
- [ ] For one-class detection, training data contains only normal instances
- [ ] Random seed set for reproducibility
- [ ] Exchangeability assumption is reasonable (or use weighted conformal)

After prediction:

- [ ] P-values are in valid range [0, 1]
- [ ] P-values have reasonable diversity (not all identical)
- [ ] No NaN or Inf values in predictions

## Further Reading

- [Conformal Inference](conformal_inference.md) - Theoretical foundations
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Best Practices](best_practices.md) - Production-ready patterns
- [Weighted Conformal](weighted_conformal.md) - Handling covariate shift