# Understanding Conformal Inference

This guide explains the theoretical foundations and practical applications of conformal inference in anomaly detection using the new nonconform API.

## What is Conformal Inference?

Conformal inference is a framework for creating prediction intervals or hypothesis tests with finite-sample validity guarantees. In the context of anomaly detection, it transforms raw anomaly scores into statistically valid p-values.

### The Problem with Traditional Anomaly Detection

Traditional anomaly detection methods output scores and require setting arbitrary thresholds:

```python
# Traditional approach - arbitrary threshold
scores = detector.decision_function(X_test)
anomalies = scores < -0.5  # Why -0.5? No statistical justification!
```

This approach has several issues:
- No statistical guarantees about error rates
- Threshold selection is often arbitrary
- No control over false positive rates
- Results are not interpretable in probabilistic terms

### The Conformal Solution

Conformal inference provides a principled way to convert scores to p-values:

```python
# Conformal approach - statistically valid p-values
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation

# Create conformal detector
strategy = Split(n_calib=0.2)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit on training data (includes automatic calibration)
detector.fit(X_train)

# Get valid p-values
p_values = detector.predict(X_test, raw=False)

# Now we can control error rates with FDR control!
from scipy.stats import false_discovery_control

# Apply Benjamini-Hochberg FDR control
fdr_corrected_pvals = false_discovery_control(p_values, method='bh')
anomalies = fdr_corrected_pvals < 0.05  # Controls FDR at 5%
```

## Mathematical Foundation

### Classical Conformal p-values

Given a scoring function $s(X)$ where higher scores indicate more anomalous behavior, and a calibration set $D_{calib} = \{X_1, \ldots, X_n\}$, the classical conformal p-value for a test instance $X_{test}$ is:

$$p_{classical}(X_{test}) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{s(X_i) \geq s(X_{test})\}}{n+1}$$

where $\mathbf{1}\{\cdot\}$ is the indicator function.

### Statistical Validity

**Key Property**: If $X_{test}$ is exchangeable with the calibration data (i.e., drawn from the same distribution), then:

$$\mathbb{P}(p_{classical}(X_{test}) \leq \alpha) \leq \alpha$$

for any $\alpha \in (0,1)$.

**Important Note**: This guarantee holds under the null hypothesis that $X_{test}$ is from the same distribution as the calibration data. For a test instance that is truly anomalous (not from the calibration distribution), this probability statement does not apply.

This means that if we declare $X_{test}$ anomalous when $p_{classical}(X_{test}) \leq 0.05$, we'll have at most a 5% false positive rate **among normal instances**. The overall false positive rate in practice depends on the proportion of normal vs. anomalous instances in your test data.

### Intuitive Understanding

The p-value answers the question: "If this test instance were actually normal, what's the probability of observing an anomaly score at least as extreme as what we observed?"

- **High p-value (e.g., 0.8)**: The test instance looks very similar to calibration data
- **Medium p-value (e.g., 0.3)**: The test instance is somewhat unusual but not clearly anomalous
- **Low p-value (e.g., 0.02)**: The test instance is very different from calibration data

## Exchangeability Assumption

### What is Exchangeability?

Exchangeability is weaker than the i.i.d. assumption. A sequence of random variables $(X_1, X_2, \ldots, X_n)$ is exchangeable if their joint distribution is invariant to permutations. Formally, for any permutation $\pi$ of $\{1, 2, \ldots, n\}$:

$$P(X_1 \leq x_1, \ldots, X_n \leq x_n) = P(X_{\pi(1)} \leq x_1, \ldots, X_{\pi(n)} \leq x_n)$$

**Key insight for conformal prediction**: Under exchangeability, if we add a new observation $X_{n+1}$ from the same distribution, then $(X_1, \ldots, X_n, X_{n+1})$ remains exchangeable. This means that $X_{n+1}$ is equally likely to have the $k$-th largest value among all $n+1$ observations for any $k \in \{1, \ldots, n+1\}$.

### When Exchangeability Holds

**Practical insight**: Exchangeability means that the order in which you observe your data points doesn't matter - there's no systematic pattern or trend that makes earlier observations systematically different from later ones.

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

**Solution**: Weighted conformal prediction uses density ratio estimation to reweight calibration data, potentially restoring valid inference under **specific types of covariate shift**. **Key limitations**:

1. **Assumption**: Requires that P(Y|X) remains constant while only P(X) changes
2. **Density ratio estimation errors**: Inaccurate weight estimation can degrade or even worsen performance
3. **High-dimensional challenges**: Density ratio estimation becomes unreliable in high dimensions or with limited data
4. **Distribution support**: Requires sufficient overlap between calibration and test distributions
5. **No guarantee**: Unlike standard conformal prediction, weighted methods may not maintain exact finite-sample guarantees when assumptions are violated

The method estimates the likelihood ratio dP_test(X)/dP_calib(X) and reweights calibration data accordingly. Success depends critically on both the validity of the covariate shift assumption and the quality of density ratio estimation.

## Practical Implementation

### Basic Setup

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func import Aggregation

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
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# 4. Fit detector (automatically handles train/calibration split)
detector.fit(X_train)

# 5. Get p-values for test data
p_values = detector.predict(X_test, raw=False)
```

### Understanding the Output

```python
# p-values are between 0 and 1
print(f"P-values range: [{p_values.min():.4f}, {p_values.max():.4f}]")

# Small p-values indicate anomalies
suspicious_indices = np.where(p_values < 0.05)[0]
print(f"Suspicious instances: {len(suspicious_indices)}")

# Very small p-values are strong evidence
very_suspicious = np.where(p_values < 0.01)[0]
print(f"Very suspicious instances: {len(very_suspicious)}")

# P-value interpretation
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

Best for large datasets where you can afford to hold out calibration data:

```python
from nonconform.strategy import Split

# Use 20% of data for calibration
strategy = Split(n_calib=0.2)

# Or use absolute number for very large datasets
strategy = Split(n_calib=1000)
```

### 2. Cross-Validation Strategy

Better utilization of data by using all samples for both training and calibration:

```python
from nonconform.strategy import CrossValidation

# 5-fold cross-validation
strategy = CrossValidation(k=5)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

### 3. Bootstrap Strategy

Provides robust estimates through resampling:

```python
from nonconform.strategy import Bootstrap

# 100 bootstrap samples with 80% sampling ratio
strategy = Bootstrap(n_bootstraps=100, resampling_ratio=0.8)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

### 4. Jackknife Strategy (Leave-One-Out)

Maximum use of small datasets:

```python
from nonconform.strategy import Jackknife

# Leave-one-out cross-validation
strategy = Jackknife()

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
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
raw_scores = detector.predict(X_test, raw=True)

# Get p-values
p_values = detector.predict(X_test, raw=False)

# Understand the relationship
import matplotlib.pyplot as plt
plt.scatter(raw_scores, p_values)
plt.xlabel('Raw Anomaly Score')
plt.ylabel('P-value')
plt.title('Score vs P-value Relationship')
plt.show()
```

### Aggregation Methods

When using ensemble strategies, you can control how multiple model outputs are combined:

```python
# Different aggregation methods
aggregation_methods = [Aggregation.MEAN, Aggregation.MEDIAN, Aggregation.MAX]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=CrossValidation(k=5),
        aggregation=agg_method,
        seed=42
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test, raw=False)

    print(f"{agg_method.value}: {(p_values < 0.05).sum()} detections")
```

**Note on p-value averaging**: The aggregation shown here averages conformal p-values from the same underlying procedure with different aggregation methods (e.g., MEAN vs. MEDIAN). This is distinct from traditional p-value combination methods and preserves conformal validity since all p-values derive from the same exchangeable framework.

### Custom Scoring Functions

For advanced users, you can create custom detectors:

```python
from pyod.models.base import BaseDetector

class CustomDetector(BaseDetector):
    """Custom anomaly detector following PyOD interface."""

    def __init__(self, contamination=0.1):
        super().__init__(contamination=contamination)

    def fit(self, X, y=None):
        # Your custom fitting logic here
        self.decision_scores_ = self._compute_scores(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        # Your custom scoring logic here
        return self._compute_scores(X)

    def _compute_scores(self, X):
        # Higher scores should indicate more anomalous behavior
        # This is a dummy implementation
        return np.random.random(len(X))

# Use with conformal detection
custom_detector = CustomDetector()
detector = ConformalDetector(
    detector=custom_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

## Performance Considerations

### Computational Complexity

Different strategies have different computational costs:

```python
import time

strategies = {
    'Split': Split(calib_size=0.2),
    'Cross-Val (5-fold)': CrossValidation(k=5),
    'Bootstrap (50)': Bootstrap(n_bootstraps=50, resampling_ratio=0.8),
    'Jackknife': Jackknife()
}

for name, strategy in strategies.items():
    start_time = time.time()

    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
        silent=True
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test, raw=False)

    elapsed = time.time() - start_time
    print(f"{name}: {elapsed:.2f}s ({(p_values < 0.05).sum()} detections)")
```

### Memory Usage

For large datasets, consider:

```python
# Use batch processing for very large test sets
import itertools

def predict_in_batches(detector, X_test, batch_size=1000):
    all_p_values = []

    for batch in itertools.batched(X_test, batch_size):
        batch_p_values = detector.predict(batch, raw=False)
        all_p_values.extend(batch_p_values)

    return np.array(all_p_values)

# Usage for large datasets
p_values = predict_in_batches(detector, X_test_large)
```

## Next Steps

- Learn about [different conformalization strategies](conformalization_strategies.md) in detail
- Understand [weighted conformal p-values](weighted_conformal.md) for handling distribution shift
- Explore [FDR control](fdr_control.md) for multiple testing scenarios
- Check out [best practices](best_practices.md) for production deployment
- Review the [troubleshooting guide](troubleshooting.md) for common issues
