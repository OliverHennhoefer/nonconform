# Choosing Calibration Strategies

This guide helps you select the optimal calibration strategy for your conformal anomaly detection task based on dataset characteristics, computational constraints, and accuracy requirements.

## Strategy Overview

Nonconform provides four calibration strategies, each with distinct trade-offs:

| Strategy | Speed | Accuracy | Data Efficiency | Best For |
|----------|-------|----------|-----------------|----------|
| **Split** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Large datasets, real-time |
| **Jackknife+** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | General purpose, balanced |
| **Cross-Validation** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Small datasets, maximum accuracy |
| **Bootstrap** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Uncertainty quantification |

## Detailed Strategy Characteristics

### Split Conformal

**When to use:**
- Large training datasets (>5,000 samples)
- Real-time or production environments requiring fast inference
- When computational resources are limited
- Initial prototyping and development

**Advantages:**
- Fastest training and inference
- Minimal memory usage
- Simple to understand and implement
- Predictable computational cost

**Disadvantages:**
- Uses only a subset of data for calibration
- May be less reliable with small datasets
- No theoretical optimality guarantees

**Configuration example:**
```python
from nonconform.strategy import Split

# For large datasets
strategy = Split(n_calib=0.2)  # Use 20% for calibration

# For fixed calibration size
strategy = Split(n_calib=2000)  # Use exactly 2000 samples
```

### Jackknife+ Conformal

**When to use:**
- Medium-sized datasets (1,000-10,000 samples)
- When you need good accuracy without excessive computation
- Production systems with moderate performance requirements
- General-purpose applications

**Advantages:**
- Uses all training data efficiently
- Provides theoretical finite-sample guarantees
- Good balance of speed and accuracy
- Automatic calibration set sizing

**Disadvantages:**
- More computationally expensive than Split
- Memory usage scales with training set size
- Cannot easily parallelize calibration

**Configuration example:**
```python
from nonconform.strategy import Jackknife

# Standard Jackknife+ (recommended)
strategy = Jackknife(plus=True)

# Regular Jackknife (less conservative)
strategy = Jackknife(plus=False)
```

### Cross-Validation Conformal

**When to use:**
- Small to medium datasets (<5,000 samples)
- When maximum data efficiency is crucial
- Research applications requiring robust results
- When you have sufficient computational budget

**Advantages:**
- Most efficient use of available data
- Provides robust calibration estimates
- Works well with limited training data
- Theoretical guarantees with finite-sample corrections

**Disadvantages:**
- Highest computational cost
- Memory intensive for large datasets
- Longer training times
- Complex implementation

**Configuration example:**
```python
from nonconform.strategy import CrossValidation

# Standard 5-fold CV+ (recommended)
strategy = CrossValidation(n_folds=5, plus=True)

# More folds for smaller datasets
strategy = CrossValidation(n_folds=10, plus=True)

# Faster alternative without plus correction
strategy = CrossValidation(n_folds=3, plus=False)
```

### Bootstrap Conformal

**When to use:**
- When uncertainty quantification is critical
- Research applications requiring statistical robustness
- Noisy or heterogeneous training data
- When computational cost is not a primary concern

**Advantages:**
- Most robust calibration under model uncertainty
- Provides distribution of calibration estimates
- Works well with complex data distributions
- Best theoretical properties

**Disadvantages:**
- Highest computational cost
- Requires careful tuning of bootstrap parameters
- Memory intensive
- Longest training times

**Configuration example:**

```python
from nonconform.strategy import Bootstrap

# Standard bootstrap (recommended starting point)
strategy = Bootstrap(n_bootstraps=50, n_calib=0.2)

# High-precision bootstrap for research
strategy = Bootstrap(n_bootstraps=200, n_calib=0.3)

# Fast bootstrap for prototyping
strategy = Bootstrap(n_bootstraps=20, n_calib=0.15)
```

## Decision Framework

### 1. Dataset Size Considerations

**Large datasets (>10,000 samples):**
- **Primary choice:** Split (fast, efficient)
- **Alternative:** Jackknife+ (if accuracy is critical)

**Medium datasets (1,000-10,000 samples):**
- **Primary choice:** Jackknife+ (balanced performance)
- **Alternative:** Cross-Validation (if maximum accuracy needed)

**Small datasets (<1,000 samples):**
- **Primary choice:** Cross-Validation (data efficient)
- **Alternative:** Bootstrap (if robustness critical)

### 2. Performance Requirements

**Real-time applications (latency <100ms):**
- Use Split conformal
- Pre-compute calibration sets where possible
- Consider caching fitted detectors

**Batch processing (latency <10s):**
- Jackknife+ or Cross-Validation
- Optimize based on accuracy requirements

**Offline analysis (no latency constraints):**
- Any strategy based on accuracy needs
- Bootstrap for maximum robustness

### 3. Accuracy vs Speed Trade-offs

**Maximum speed (production systems):**
```python
# Fastest configuration
strategy = Split(n_calib=1000)  # Fixed size for predictable performance
```

**Balanced (general applications):**
```python
# Good accuracy with reasonable speed
strategy = Jackknife(plus=True)
```

**Maximum accuracy (research/critical applications):**
```python
# Most robust but slower
strategy = CrossValidation(n_folds=10, plus=True)
```

## Advanced Considerations

### Data Distribution Properties

**Exchangeable data (IID assumption holds):**
- All strategies work well
- Choose based on computational constraints

**Non-exchangeable data (distribution shift):**
- Consider weighted conformal detection
- Bootstrap strategy may provide additional robustness
- Monitor calibration performance over time

**Heterogeneous data (mixed distributions):**
- Bootstrap conformal recommended
- Cross-validation as alternative
- Avoid Split with very diverse training sets

### Computational Resource Planning

**Memory constraints:**
- Split: O(n_calib) memory usage
- Jackknife+: O(n_train) memory usage
- Cross-Validation: O(n_train × n_folds) memory usage
- Bootstrap: O(n_train × n_bootstraps) memory usage

**CPU considerations:**
- Split: Single model training
- Jackknife+: n_train + 1 model trainings
- Cross-Validation: n_folds model trainings
- Bootstrap: n_bootstraps model trainings

## Strategy Migration Guide

### From Research to Production

1. **Development phase:** Use Cross-Validation for robust results
2. **Validation phase:** Compare with Jackknife+ for speed assessment
3. **Production phase:** Deploy with Split for optimal performance
4. **Monitoring phase:** Validate that Split maintains required accuracy

### Handling Performance Degradation

If you observe degraded performance after strategy changes:

1. **Check calibration set size:** Ensure adequate samples for reliable calibration
2. **Validate data assumptions:** Verify exchangeability hasn't changed
3. **Monitor drift:** Use weighted conformal if distribution shift detected
4. **Adjust parameters:** Tune strategy-specific parameters

## Common Pitfalls

### Split Conformal
- **Don't:** Use with very small datasets (<500 samples)
- **Don't:** Use fixed small calibration sets with varying dataset sizes
- **Do:** Use proportional calibration sizing for consistency

### Jackknife+ Conformal
- **Don't:** Use with extremely large datasets if memory is constrained
- **Don't:** Forget that it requires n+1 model fits
- **Do:** Enable parallel processing where available

### Cross-Validation Conformal
- **Don't:** Use too many folds with small datasets (overfitting risk)
- **Don't:** Use without plus correction in critical applications
- **Do:** Balance n_folds with computational budget

### Bootstrap Conformal
- **Don't:** Use too few bootstraps (<20) for robust estimates
- **Don't:** Ignore bootstrap variance in interpretation
- **Do:** Monitor convergence of bootstrap estimates

## Benchmarking Your Choice

Always validate your strategy choice with performance metrics:

```python
from nonconform.utils.stat import false_discovery_rate, statistical_power

# Compare strategies on your data
strategies = {
    'Split': Split(n_calib=0.2),
    'Jackknife+': Jackknife(plus=True),
    'CrossVal': CrossValidation(n_folds=5, plus=True)
}

for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=your_detector,
        strategy=strategy,
        seed=42
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test)

    # Evaluate performance
    decisions = false_discovery_control(p_values) <= 0.1
    fdr = false_discovery_rate(y_test, decisions)
    power = statistical_power(y_test, decisions)

    print(f"{name}: FDR={fdr:.3f}, Power={power:.3f}")
```

Choose the strategy that best meets your specific requirements for FDR control, statistical power, and computational performance.
