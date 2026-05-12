# Choosing Calibration Strategies

This guide helps you choose a calibration strategy based on data size, runtime,
memory, and how clean a validity story you need. The recommendations are
starting points, not universal optima.

## Strategy Overview

nonconform provides one simple split baseline and a family of resampling
strategies for cases where a holdout calibration split would waste too much
data:

| Strategy | Speed | Data Efficiency | Validity Story | Best For |
|---|---|---|---|---|
| **Split** | High | Medium | Cleanest | Large datasets, production baselines |
| **CV+** | Medium | High | Resampling-based | Practical small-data default |
| **Jackknife+** | Low | Very high | Resampling-based | Very small datasets |
| **JackknifeBootstrap (JaB+)** | Low | High | Looser resampling bound | Bootstrap stability |

> **Guarantee note:** Split conformal is the cleanest strict finite-sample
> baseline. Resampling strategies such as cross-conformal, CV+, Jackknife+, and
> JaB+ use data more efficiently and often work well in practice, but their
> guarantees are weaker, approximate, asymptotic, or looser depending on the
> method. `mode="plus"` is the validity-oriented default for these families;
> `mode="single_model"` is lighter but weakens the validity story further.

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
from nonconform import Split

# For large datasets
strategy = Split(n_calib=0.2)  # Use 20% for calibration

# For fixed calibration size
strategy = Split(n_calib=2000)  # Use exactly 2000 samples
```

### Data-Efficient Resampling

**When to use:**
- Small to medium datasets where a fixed calibration holdout is too costly
- Applications where every observation should help train at least one model
- Research workflows where you can spend extra computation for smoother results
- Production workflows where the memory and latency costs are acceptable

**Advantages:**
- Avoids permanently reserving a calibration-only subset
- Lets each observation contribute through folds, leave-one-out fits, or
  bootstrap out-of-bag structure
- Often improves practical power when data is scarce
- `mode="plus"` gives the most defensible resampling option in this package

**Disadvantages:**
- More computationally expensive than Split
- Memory usage can grow with folds, leave-one-out models, or bootstraps
- Guarantees are weaker, approximate, asymptotic, or looser than the clean
  split-conformal baseline, depending on the method
- Method choice depends on dataset size and compute budget

**Configuration examples:**
```python
from nonconform import CrossValidation, JackknifeBootstrap

# Practical default for limited data
strategy = CrossValidation(k=5, mode="plus")

# Leave-one-out variant for very small datasets
strategy = CrossValidation.jackknife(mode="plus")

# Bootstrap variant for stability analysis
strategy = JackknifeBootstrap(n_bootstraps=100, mode="plus")
```

**How to choose inside the family:**

| Method | Good First Use | Watch Out For |
|--------|----------------|---------------|
| CV+ | Limited data with practical compute | More folds cost more model fits |
| Jackknife+ | Very small data | Leave-one-out fitting can be expensive |
| JaB+ | Bootstrap stability or noisy data | Too few bootstraps can be unstable |

## Decision Framework

The thresholds below are practical defaults. Use labeled validation data when
available and compare strategies on empirical FDR, power, runtime, and memory.

### 1. Dataset Size Considerations

**Large datasets (>10,000 samples):**
- **Primary choice:** Split (fast, efficient)
- **Alternative:** JackknifeBootstrap (if speed is not the top priority)

**Medium datasets (1,000-10,000 samples):**
- **Primary choice:** JackknifeBootstrap (balanced robustness and practicality)
- **Alternative:** Jackknife+ (if you want lower compute than larger-bootstrap setups)

**Small datasets (<1,000 samples):**
- **Primary choice:** Jackknife+
- **Alternative:** Jackknife (for the smallest datasets)

### 2. Performance Requirements

**Real-time applications (latency <100ms):**
- Use Split conformal
- Pre-compute calibration sets where possible
- Consider caching fitted detectors

**Batch processing (latency <10s):**
- Jackknife+ or JackknifeBootstrap
- Optimize based on accuracy requirements

**Offline analysis (no latency constraints):**
- Any strategy based on accuracy needs
- JackknifeBootstrap for maximum robustness

### 3. Accuracy vs Speed Trade-offs

**Maximum speed (production systems):**
```python
# Fastest configuration
strategy = Split(n_calib=1000)  # Fixed size for predictable performance
```

**Balanced (general applications):**
```python
# Good robustness with practical defaults
strategy = JackknifeBootstrap(n_bootstraps=100)
```

**Maximum robustness checks (research/high-rigor applications):**
```python
# More resampling stability, but slower
strategy = JackknifeBootstrap(n_bootstraps=200)
```

## Advanced Considerations

### Data Distribution Properties

**Exchangeable data (IID assumption holds):**
- All strategies work well
- Choose based on computational constraints

**Non-exchangeable data (distribution shift):**
- Consider weighted conformal detection only when the shift is plausibly
  covariate shift with support overlap
- JackknifeBootstrap strategy may provide additional robustness
- Monitor calibration performance over time

**Heterogeneous data (mixed distributions):**
- JackknifeBootstrap recommended
- Jackknife+ as alternative
- Avoid Split with very diverse training sets

### Computational Resource Planning

**Memory constraints:**
- Split: O(n_calib) memory usage
- Jackknife+: O(n_train) memory usage
- Cross-Validation: O(k × n_test) inference peak; O(k) stored models + O(n_train) calibration scores
- JackknifeBootstrap: O(n_train x n_bootstraps) memory usage (includes permanent `_oob_mask` storage)

**CPU considerations:**
- Split: Single model training
- Jackknife+: n_train + 1 model trainings
- Cross-Validation: n_folds model trainings
- JackknifeBootstrap: n_bootstraps model trainings

## Strategy Transition Guide

### From Research to Production

1. **Development phase:** Use JackknifeBootstrap for robust results
2. **Validation phase:** Compare with Jackknife+ for speed assessment
3. **Production phase:** Deploy with Split when latency, memory, and simple
   validation are the priorities
4. **Monitoring phase:** Validate that Split maintains required accuracy

### Handling Performance Degradation

If you observe degraded performance after strategy changes:

1. **Check calibration set size:** Ensure adequate samples for reliable calibration
2. **Validate data assumptions:** Verify exchangeability hasn't changed
3. **Monitor drift:** Use weighted conformal only when detected drift matches
   the covariate-shift assumptions
4. **Adjust parameters:** Tune strategy-specific parameters

## Common Pitfalls

### Split Conformal
- **Don't:** Use with very small datasets (<500 samples)
- **Don't:** Use fixed small calibration sets with varying dataset sizes
- **Do:** Use proportional calibration sizing for consistency

### Resampling Strategies
- **Don't:** Use too many folds with small datasets (overfitting risk)
- **Don't:** Treat `mode="single_model"` as equivalent to plus-style resampling
- **Don't:** Forget that Jackknife+ requires one fit per observation
- **Don't:** Use too few bootstraps (<20) for robust estimates
- **Do:** Balance folds, leave-one-out fits, or bootstraps against your compute budget
- **Do:** Monitor bootstrap stability when using JaB+

## Benchmarking Your Choice

Always validate your strategy choice with performance metrics:

```python
from nonconform import ConformalDetector, CrossValidation, JackknifeBootstrap, Split
from nonconform.metrics import false_discovery_rate, statistical_power

# Compare strategies on your data
strategies = {
    "Split": Split(n_calib=0.2),
    "CV+": CrossValidation(k=5, mode="plus"),
    "Jackknife+": CrossValidation.jackknife(mode="plus"),
    "JaB+": JackknifeBootstrap(n_bootstraps=100, mode="plus"),
}

for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=your_detector,
        strategy=strategy,
        seed=42
    )
    detector.fit(X_train)
    decisions = detector.select(X_test, alpha=0.1)

    # Evaluate FDR-controlled decisions
    fdr = false_discovery_rate(y_test, decisions)
    power = statistical_power(y_test, decisions)

    print(f"{name}: FDR={fdr:.3f}, Power={power:.3f}")
```

Choose the strategy that best meets your requirements for FDR control,
statistical power, runtime, and memory. When in doubt, keep Split as a baseline:
it is easier to reason about, and it makes assumption failures easier to spot.

## References

For the statistical background behind these recommendations, see
[Conformalization Strategies](conformalization_strategies.md#references).
