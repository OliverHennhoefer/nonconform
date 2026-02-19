# Understanding Conformal Inference

Learn the theoretical foundations of conformal inference for anomaly detection.

!!! abstract "TL;DR"
    **Conformal inference converts anomaly scores into p-values with statistical guarantees.**

    - **The problem**: Traditional detectors output arbitrary scores with no principled threshold
    - **The solution**: Compare each test point's score against a calibration set to compute a p-value
    - **The guarantee**: If a point is truly normal, its p-value is uniformly distributed—so a threshold of 0.05 gives exactly 5% false positives
    - **Key assumption**: Training and test data must be **exchangeable** (roughly: drawn from the same distribution)
    - **For distribution shift**: Use weighted conformal prediction to adjust for differences between training and test distributions

## What is Conformal Inference?

Conformal inference is a framework for creating prediction intervals or hypothesis tests with finite-sample validity guarantees [[Vovk et al., 2005](#references); [Shafer & Vovk, 2008](#references)]. In the context of anomaly detection, it transforms raw anomaly scores into statistically valid p-values [[Bates et al., 2023](#references)].

### The Problem with Traditional Anomaly Detection

Traditional anomaly detectors output scores and require arbitrary thresholds:

```python
# Traditional approach - arbitrary threshold
scores = detector.decision_function(X_test)
anomalies = scores < -0.5  # Why -0.5? No statistical justification!
```

This approach has several issues:
- No error rate guarantees
- Arbitrary threshold selection
- No false positive control
- Non-probabilistic output

### The Conformal Solution

Conformal inference provides a principled way to convert scores to p-values:

```python
# Conformal approach - statistically valid p-values
from nonconform import ConformalDetector, Split

from scipy.stats import false_discovery_control

# Create conformal detector
strategy = Split(n_calib=0.2)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    seed=42
)

# Fit on training data (includes automatic calibration) and get p-values
p_values = detector.fit(X_train).compute_p_values(X_test)

# Apply Benjamini-Hochberg FDR control
fdr_corrected_pvals = false_discovery_control(p_values, method='bh')
anomalies = fdr_corrected_pvals < 0.05  # Controls FDR at 5%
```

`fit(...)` remains the default one-call workflow: train + calibrate together.
If your base model is already trained in a separate pipeline stage, you can
calibrate separately with `detector.calibrate(X_calib)` (currently split-style
detached calibration with `Split` strategy).

## Mathematical Foundation

### Classical Conformal p-values

Given a scoring function $s(X)$ where higher scores indicate more anomalous behavior, and a calibration set $D_{calib} = \{X_1, \ldots, X_n\}$, the classical conformal p-value for a test instance $X_{test}$ is:

$$p_{classical}(X_{test}) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{s(X_i) \geq s(X_{test})\}}{n+1}$$

where $\mathbf{1}\{\cdot\}$ is the indicator function.

**In plain English**: The p-value is the fraction of calibration points that have scores at least as extreme as the test point. If 5 out of 100 calibration points have higher scores than your test point, the p-value is (1+5)/(100+1) ≈ 0.06. The "+1" terms ensure the p-value is never exactly 0 and accounts for the test point itself.

### Statistical Validity

!!! tip "Key Property"
    If $X_{test}$ is exchangeable with the calibration data (i.e., drawn from the same distribution), then [[Vovk et al., 2005](#references)]:

    $$\mathbb{P}(p_{classical}(X_{test}) \leq \alpha) \leq \alpha$$

    for any $\alpha \in (0,1)$.

!!! warning "Statistical Assumption"
    This guarantee holds under the null hypothesis that $X_{test}$ comes from the same distribution as calibration data. For truly anomalous instances (not from the calibration distribution), this probability statement does not apply.

This means that if we declare $X_{test}$ anomalous when $p_{classical}(X_{test}) \leq 0.05$, we'll have at most a 5% false positive rate **among normal instances**. The overall false positive rate in practice depends on the proportion of normal vs. anomalous instances in your test data.

### Intuitive Understanding

The p-value answers: "If this instance were normal, what's the probability of a score this extreme or higher?"

- **High p-value (e.g., 0.8)**: The test instance looks very similar to calibration data
- **Medium p-value (e.g., 0.3)**: The test instance is somewhat unusual but not clearly anomalous
- **Low p-value (e.g., 0.02)**: The test instance is very different from calibration data

### Randomized/Smoothed P-values

Building on the classical conformal framework, [[Jin & Candès, 2023](#references)] introduced randomized smoothing to handle ties in the calibration scores. The randomized conformal p-value is:

$$p_{rand}(X_{test}) = \frac{|\{i: s(X_i) > s(X_{test})\}| + U \cdot (|\{i: s(X_i) = s(X_{test})\}| + 1)}{n+1}$$

where $U \sim \text{Uniform}[0,1]$ is a random tie-breaker. The "+1" accounts for the test point itself (with weight 1 in the unweighted case).

**Why randomize?** Classical p-values are limited to discrete values $k/(n+1)$, creating a resolution floor. With many tied scores, this can severely limit the granularity of p-values. Randomized smoothing eliminates this floor by spreading tied observations across the [0,1] interval.

```python
from nonconform import Empirical

# Classical (default)
estimation = Empirical()

# Randomized smoothing
estimation = Empirical(tie_break="randomized")
```

!!! info "`Empirical` `tie_break` parameter"
    - Default when omitted: `tie_break="classical"`
    - Valid string values: `"classical"` and `"randomized"` (only)
    - Enum equivalents: `TieBreakMode.CLASSICAL` and `TieBreakMode.RANDOMIZED`
    - `None` is not a valid value

!!! warning "Small Calibration Sets"
    With small calibration sets, randomized smoothing can produce anti-conservative p-values that may work against the nominal FDR level. Consider using the classical formula or the `Probabilistic()` estimator in such cases.

!!! tip "Alternative: Probabilistic Estimation"
    The `Probabilistic()` estimator uses kernel density estimation (KDE) to produce continuous p-values. This addresses both the resolution collapse of classical discrete p-values and the potential variance increase from randomized smoothing. Note that this trades the finite-sample guarantee of conformal p-values for an asymptotic guarantee.

## Exchangeability Assumption

### What is Exchangeability?

Exchangeability is weaker than the i.i.d. assumption [[Vovk et al., 2005](#references)]. A sequence of random variables $(X_1, X_2, \ldots, X_n)$ is exchangeable if their joint distribution is invariant to permutations. Formally, for any permutation $\pi$ of $\{1, 2, \ldots, n\}$:

$$P(X_1 \leq x_1, \ldots, X_n \leq x_n) = P(X_{\pi(1)} \leq x_1, \ldots, X_{\pi(n)} \leq x_n)$$

**In plain English**: Exchangeability means "the order doesn't matter." If you shuffled your data points randomly, the statistical properties would be the same. This is weaker than requiring the data to be independent—it just requires that no observation is systematically different from the others.

**Key insight for conformal prediction**: Under exchangeability, if we add a new observation $X_{n+1}$ from the same distribution, then $(X_1, \ldots, X_n, X_{n+1})$ remains exchangeable [[Angelopoulos & Bates, 2023](#references)]. This means that $X_{n+1}$ is equally likely to have the $k$-th largest value among all $n+1$ observations for any $k \in \{1, \ldots, n+1\}$.

### When Exchangeability Holds

**Practical insight**: Exchangeability means observation order doesn't matter—no systematic differences between earlier and later observations.

**Conditions for validity**:
- Training and test data come from the same source/process
- No systematic changes over time (stationarity)
- Same measurement conditions and feature distributions
- No covariate shift between calibration and test phases

Under exchangeability, standard conformal p-values provide exact finite-sample false positive rate control: for any significance level $\alpha$, the probability that a normal instance receives a p-value ≤ $\alpha$ is at most $\alpha$. This enables principled anomaly detection with known error rates and valid FDR control procedures.

### When Exchangeability is Violated

**Common violations**:
- **Covariate shift**: Test data features have different distributions than training
- **Temporal drift**: Data characteristics change over time
- **Domain shift**: Different measurement conditions, sensors, or environments
- **Selection bias**: Non-random sampling between training and test phases

**Statistical consequence**: When exchangeability fails, standard conformal p-values lose their coverage guarantees and may become systematically miscalibrated.

**Solution**: Weighted conformal prediction uses density ratio estimation to reweight calibration data, restoring validity under certain covariate shifts [[Jin & Candès, 2023](#references); [Tibshirani et al., 2019](#references)]. **Key limitations**:

1. **Assumption**: Requires that P(Y|X) remains constant while only P(X) changes
2. **Density ratio estimation errors**: Inaccurate weight estimation can degrade or even worsen performance
3. **High-dimensional challenges**: Density ratio estimation becomes unreliable in high dimensions or with limited data
4. **Distribution support**: Requires sufficient overlap between calibration and test distributions
5. **No guarantee**: Unlike standard conformal prediction, weighted methods may not maintain exact finite-sample guarantees when assumptions are violated

The method estimates dP_test(X)/dP_calib(X) and reweights accordingly. Success depends on both valid covariate shift assumptions and accurate density ratio estimation.

## Practical Implementation

### Basic Setup

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from nonconform import ConformalDetector, Split


# 1. Prepare your data
X_train = load_normal_training_data()  # Normal data for training and calibration
X_test = load_test_data()  # Data to be tested

# 2. Create base detector
base_detector = IsolationForest(random_state=42)

# 3. Create conformal detector with strategy
strategy = Split(n_calib=0.2)  # 20% for calibration
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    seed=42
)

# 4. Fit detector and get p-values
p_values = detector.fit(X_train).compute_p_values(X_test)
```

### Detached Calibration with a Pre-Trained Detector

When model training and conformal calibration happen in separate steps, train
the base detector first, then call `calibrate(...)` on dedicated calibration data:

```python
from sklearn.ensemble import IsolationForest
from nonconform import ConformalDetector, Split

base_detector = IsolationForest(random_state=42)
base_detector.fit(X_fit)

detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.2),
    aggregation="median",
    seed=42
)
detector.calibrate(X_calib)
p_values = detector.compute_p_values(X_test)
```

### Understanding the Output

```python
from scipy.stats import false_discovery_control

# p-values are between 0 and 1
print(f"P-values range: [{p_values.min():.4f}, {p_values.max():.4f}]")

# For actual anomaly detection, always apply FDR control
adjusted_p_values = false_discovery_control(p_values, method='bh')
discoveries = adjusted_p_values < 0.05
print(f"FDR-controlled discoveries: {discoveries.sum()}")

# Individual p-value interpretation (for understanding, not decision-making)
# Note: Use FDR-controlled decisions for actual anomaly detection
for i, p_val in enumerate(p_values[:5]):
    if p_val < 0.01:
        print(f"Instance {i}: p={p_val:.4f} - Strong evidence of anomaly")
    elif p_val < 0.05:
        print(f"Instance {i}: p={p_val:.4f} - Moderate evidence of anomaly")
    elif p_val < 0.1:
        print(f"Instance {i}: p={p_val:.4f} - Weak evidence of anomaly")
    else:
        print(f"Instance {i}: p={p_val:.4f} - Consistent with normal behavior")
```

## Strategies for Different Scenarios

### 1. Split Strategy

Best for large datasets with sufficient calibration data:

```python
from nonconform import Split

# Use 20% of data for calibration
strategy = Split(n_calib=0.2)

# Or use absolute number for very large datasets
strategy = Split(n_calib=1000)
```

### 2. Cross-Validation Strategy

Uses all samples for both training and calibration:

```python
from nonconform import CrossValidation


# 5-fold cross-validation
strategy = CrossValidation(k=5)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    seed=42
)
```

!!! info "CrossValidation `mode` parameter"
    `mode` controls model retention behavior (how many fitted models are kept for inference), not which statistical strategy is used.

    - Default when omitted: `mode="plus"`
    - Valid string values: `"plus"` and `"single_model"` (only)
    - Enum equivalents: `ConformalMode.PLUS` and `ConformalMode.SINGLE_MODEL`
    - `single_model` means "fit one final model after calibration" (it is not a separate Jackknife/CV method)

### 3. Jackknife+-after-Bootstrap (JaB+) Strategy

Provides robust estimates through resampling:

```python
from nonconform import JackknifeBootstrap


# 50 bootstrap samples
strategy = JackknifeBootstrap(n_bootstraps=50)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation="median",
    seed=42
)
```

!!! info "JaB+ `mode` parameter"
    `JackknifeBootstrap` uses the same `mode` options and defaults as `CrossValidation`:
    - Default when omitted: `mode="plus"`
    - Valid values: `"plus"` and `"single_model"` (or `ConformalMode.PLUS` / `ConformalMode.SINGLE_MODEL`)

!!! info "Leave-One-Out (Jackknife)"
    For leave-one-out cross-validation, use the `CrossValidation.jackknife()` factory method which handles this automatically. Alternatively, use `CrossValidation(k=n)` where `n` is your dataset size.

    ```python
    # Default is mode="plus" (Jackknife+)
    strategy = CrossValidation.jackknife()

    # Explicit options (the only valid mode strings):
    strategy = CrossValidation.jackknife(mode="plus")          # Jackknife+
    strategy = CrossValidation.jackknife(mode="single_model")  # Standard Jackknife
    ```

## Common Pitfalls and Solutions

### 1. Data Leakage
- **Problem**: Using contaminated calibration data invalidates statistical guarantees
- **Solution**: Ensure training data contains only verified normal samples
- **Key**: Never train on data containing known anomalies

### 2. Insufficient Calibration Data
- **Problem**: Too few calibration samples lead to coarse p-values
- **Solution**: Use jackknife strategy for small datasets or increase calibration set size
- **Rule of thumb**: Minimum 50-100 calibration samples for reasonable p-value resolution

### 3. Distribution Shift
- **Problem**: Test distribution differs from training distribution violates exchangeability
- **Solution**: Use weighted conformal prediction to handle covariate shift
- **Detection**: Monitor p-value distributions for systematic bias

### 4. Multiple Testing
- **Problem**: Testing many instances inflates false positive rate
- **Solution**: Apply Benjamini-Hochberg FDR control instead of raw thresholding
- **Best practice**: Always use `scipy.stats.false_discovery_control` for multiple comparisons

### 5. Improper Thresholding
- **Problem**: Using simple p-value thresholds without FDR control
- **Solution**: Apply proper multiple testing correction for all anomaly detection scenarios
- **Implementation**: Use `false_discovery_control(p_values, method='bh')` before thresholding

## Advanced Topics

### Raw Scores vs P-values

You can get both raw anomaly scores and p-values:

```python
# Get raw aggregated anomaly scores
raw_scores = detector.score_samples(X_test)

# Get p-values
p_values = detector.compute_p_values(X_test)

# Understand the relationship
import matplotlib.pyplot as plt
plt.scatter(raw_scores, p_values)
plt.xlabel('Raw Anomaly Score')
plt.ylabel('P-value')
plt.title('Score vs P-value Relationship')
plt.show()
```

For pandas-native workflows, outputs preserve the input index automatically:

```python
X_test_df = pd.DataFrame(X_test, index=my_index)
p_values = detector.compute_p_values(X_test_df)   # pd.Series indexed like X_test_df
raw_scores = detector.score_samples(X_test_df)    # pd.Series indexed like X_test_df
```

### Aggregation Methods

When using ensemble strategies, you can control how multiple model outputs are combined:

```python
# Different aggregation methods
from scipy.stats import false_discovery_control


aggregation_methods = [
    "mean",
    "median",
    "maximum",
]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=CrossValidation(k=5),
        aggregation=agg_method,
        seed=42
    )
    detector.fit(X_train)
    p_values = detector.compute_p_values(X_test)

    # Apply FDR control before counting discoveries
    adjusted = false_discovery_control(p_values, method='bh')
    discoveries = (adjusted < 0.05).sum()
    print(f"{agg_method.value}: {discoveries} discoveries")
```

**Note**: Aggregation is applied to the raw anomaly scores coming from each fold/bootstrapped detector, and the combined score is then converted to a single conformal p-value. It does *not* merge already-computed p-values. Validity is preserved because every aggregated score still comes from the same exchangeable procedure.

### Custom Scoring Functions

Any detector implementing the `AnomalyDetector` protocol works with nonconform:

```python
from typing import Any, Self
import numpy as np


class CustomDetector:
    """Custom anomaly detector implementing AnomalyDetector protocol."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        # Your custom fitting logic here
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Higher scores should indicate more anomalous behavior
        return np.random.default_rng(self.random_state).random(len(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"random_state": self.random_state}

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Use with conformal detection
custom_detector = CustomDetector(random_state=42)
detector = ConformalDetector(
    detector=custom_detector,
    strategy=strategy,
    aggregation="median",
    score_polarity="higher_is_anomalous",
    seed=42
)
```

`score_polarity` controls how detector scores are interpreted before
conformalization. Valid values are `"higher_is_anomalous"`,
`"higher_is_normal"`, and `"auto"` (or omit it).

If omitted, known sklearn normality detector families default to
`"higher_is_normal"`, while PyOD and custom detectors outside recognized
families default to `"higher_is_anomalous"`.

Use `"auto"` for strict detector-family validation (raises for custom
detectors outside recognized families).

See [Detector Compatibility](detector_compatibility.md) for more details on implementing custom detectors.

## Performance Considerations

### Computational Complexity

Different strategies have different computational costs:

```python
import time
from nonconform import CrossValidation, JackknifeBootstrap, Split


strategies = {
    'Split': Split(n_calib=0.2),
    'Cross-Val (5-fold)': CrossValidation(k=5),
    'JaB+ (50)': JackknifeBootstrap(n_bootstraps=50),
}

for name, strategy in strategies.items():
    start_time = time.time()

    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation="median",
        seed=42,
    )
    detector.fit(X_train)
    p_values = detector.compute_p_values(X_test)

    # Apply FDR control
    adjusted = false_discovery_control(p_values, method='bh')
    discoveries = (adjusted < 0.05).sum()

    elapsed = time.time() - start_time
    print(f"{name}: {elapsed:.2f}s ({discoveries} discoveries)")
```

### Memory Usage

For large datasets, consider:

```python
# Use batch processing for very large test sets
import itertools
import numpy as np

def predict_in_batches(detector, X_test, batch_size=1000):
    all_p_values = []

    for batch in itertools.batched(X_test, batch_size):
        batch_p_values = detector.compute_p_values(batch)
        all_p_values.extend(batch_p_values)

    return np.array(all_p_values)

# Usage for large datasets
p_values = predict_in_batches(detector, X_test_large)
```

## References

### Foundational Conformal Prediction

- **Vovk, V., Gammerman, A., & Shafer, G. (2005)**. *Algorithmic Learning in a Random World*. Springer. [The foundational book on conformal prediction theory and exchangeability]

- **Shafer, G., & Vovk, V. (2008)**. *A Tutorial on Conformal Prediction*. Journal of Machine Learning Research, 9, 371-421. [Accessible introduction to conformal prediction]

### Conformal Anomaly Detection

- **Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023)**. *Testing for Outliers with Conformal p-values*. The Annals of Statistics, 51(1), 149-178. [Application of conformal prediction to anomaly detection with finite-sample guarantees]

- **Angelopoulos, A. N., & Bates, S. (2023)**. *Conformal Prediction: A Gentle Introduction*. Foundations and Trends in Machine Learning, 16(4), 494-591. [Comprehensive modern introduction to conformal prediction]

### Weighted Conformal Inference

- **Jin, Y., & Candès, E. J. (2023)**. *Model-free Selective Inference Under Covariate Shift via Weighted Conformal p-values*. Biometrika, 110(4), 1090-1106. arXiv:2307.09291. [Weighted conformal methods for handling distribution shift]

- **Tibshirani, R. J., Barber, R. F., Candes, E., & Ramdas, A. (2019)**. *Conformal Prediction Under Covariate Shift*. Advances in Neural Information Processing Systems, 32. arXiv:1904.06019. [Early work on conformal prediction with covariate shift]

### Additional Resources

- **Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021)**. *Predictive Inference with the Jackknife+*. The Annals of Statistics, 49(1), 486-507. [Jackknife+ method for efficient conformal prediction]

- **Benjamini, Y., & Hochberg, Y. (1995)**. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B, 57(1), 289-300. [FDR control methodology used in multiple testing]

## Next Steps

- Learn about [different conformalization strategies](conformalization_strategies.md) in detail
- Understand [weighted conformal p-values](weighted_conformal.md) for handling distribution shift
- Explore [FDR control](fdr_control.md) for multiple testing scenarios
- Check out [best practices](best_practices.md) for production deployment
- Review the [troubleshooting guide](troubleshooting.md) for common issues
- See [input validation](input_validation.md) for parameter constraints and error handling
