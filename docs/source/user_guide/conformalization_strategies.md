# Conformalization Strategies

Calibration strategies with trade-offs between efficiency and robustness.

## Quick Decision Guide

| Dataset Size | Speed Priority | Recommendation |
|-------------|----------------|----------------|
| Large (>5,000) | Yes | `Split(n_calib=0.2)` |
| Large (>5,000) | No | `JackknifeBootstrap(n_bootstraps=100)` |
| Medium (500-5,000) | Any | `JackknifeBootstrap(n_bootstraps=100)` |
| Small (<500) | Any | `CrossValidation.jackknife()` |

For very small datasets, use `CrossValidation.jackknife()` when you need strict finite-sample guarantees.
If you need smoother p-values, consider `Probabilistic()` (KDE-based), noting this trades strict finite-sample guarantees for asymptotic behavior.
For CV/Jackknife/Bootstrap-style conformalization, strict finite-sample theoretical guarantees are tied to the `"plus"` variants (`CV+`, `Jackknife+`, `JaB+`).
`mode="single_model"` can perform similarly in practice and is lighter at inference time, but it does not provide the same strict guarantees.

For detailed guidance, see [Choosing Strategies](choosing_strategies.md).

---

## Available Strategies

### Split Strategy

Simple train/calibration split. Fast and straightforward.

```python
from nonconform import Split

# Use 30% of data for calibration
strategy = Split(n_calib=0.3)

# Use fixed number of samples for calibration
strategy = Split(n_calib=1000)
```

**Characteristics:**
- **Fastest** computation
- **Simplest** implementation
- **Least robust** for small datasets
- **Memory efficient**

### Cross-Validation Strategy

K-fold cross-validation for robust calibration using all data.

```python
from nonconform import CrossValidation

# 5-fold cross-validation with one final model kept for inference
strategy = CrossValidation(k=5, mode="single_model")

# Plus mode keeps fold models for plus-style inference (recommended)
strategy = CrossValidation(k=5, mode="plus")
```

!!! info "`mode` semantics"
    For `CrossValidation` (including `CrossValidation.jackknife(...)`) and `JackknifeBootstrap`:
    - Default when omitted: `mode="plus"`
    - Valid values: `"plus"` and `"single_model"` (or `ConformalMode.PLUS` / `ConformalMode.SINGLE_MODEL`)
    - `mode="plus"`: keeps per-fold/per-bootstrap models for plus-style inference
    - `mode="single_model"`: still calibrates via folds/bootstraps, then fits one final model on all training data for inference
    - `mode="single_model"` can weaken conformal validity; use `mode="plus"` when validity is the priority

**Characteristics:**
- **Most robust** calibration
- **Uses all data** for both training and calibration
- **Higher computational cost**
- **Useful alternative** when deterministic fold-based calibration is preferred

### JaB+ Strategy (Jackknife+-after-Bootstrap)

Bootstrap resampling with Jackknife+ for robust calibration [[Kim et al., 2020](#references)].

```python
from nonconform import JackknifeBootstrap

# Typical JaB+ starting point (100+ bootstraps recommended)
strategy = JackknifeBootstrap(n_bootstraps=100)

# Higher precision with more bootstraps
strategy = JackknifeBootstrap(n_bootstraps=200)
```

**Characteristics:**
- **Flexible ensemble** size
- **Uncertainty quantification**
- **Robust to outliers**
- **Configurable computational cost**
- **Typically recommended:** 100+ bootstraps for stable behavior

### Jackknife Strategy

Leave-one-out cross-validation for maximum data utilization [[Barber et al., 2021](#references)].

```python
from nonconform import CrossValidation

# Standard jackknife with one final inference model
strategy = CrossValidation.jackknife(mode="single_model")

# Jackknife+ keeps leave-one-out models for plus-style inference
strategy = CrossValidation.jackknife(mode="plus")
```

**Characteristics:**
- **Maximum data utilization**
- **Computationally intensive**
- **Best for very small datasets**
- **Provides individual sample influence**

## Strategy Selection Guide

| Dataset Size | Computational Budget | Recommendation |
|-------------|---------------------|----------------|
| Large (>5,000) | Low | Split |
| Large (>5,000) | High | JackknifeBootstrap |
| Medium (500-5,000) | Any | JackknifeBootstrap |
| Small (<500) | Any | Jackknife |

## Mode Semantics

CrossValidation and JackknifeBootstrap strategies support `"plus"` mode for stronger conformal validity behavior in anomaly detection workflows [[Barber et al., 2021](#references)]:

```python
# Enable plus mode for CV strategies
strategy = CrossValidation(k=5, mode="plus")
strategy = CrossValidation.jackknife(mode="plus")
strategy = JackknifeBootstrap(n_bootstraps=100, mode="plus")
```

**`mode="plus"` provides:**
- Higher statistical efficiency in theory [[Barber et al., 2021](#references)]
- Better finite-sample properties
- Slightly higher computational cost
- Strict finite-sample/theoretical guarantees for these strategy families

The "plus" suffix (e.g., Jackknife+, CV+) indicates a refined variant that is typically preferred when strict finite-sample validity is the priority.

**`mode="single_model"` provides:**
- Lower inference-time memory footprint
- One final detector trained on full data for inference
- Can be close to `mode="plus"` in practice for some datasets
- No strict finite-sample/theoretical guarantee comparable to `"plus"`

## Performance Comparison

| Strategy | Training Time | Memory Usage | Calibration Quality |
|----------|---------------|--------------|-------------------|
| Split | Fast | Low | Good |
| CrossValidation | Medium | Medium | Excellent |
| JackknifeBootstrap | Medium-High | Medium-High | Excellent |
| Jackknife (LOO) | Slow | High | Excellent |

## Integration with Detectors

All strategies work with any conformal detector:

```python
from nonconform import ConformalDetector, CrossValidation, JackknifeBootstrap, logistic_weight_estimator
from pyod.models.lof import LOF

# Standard conformal with cross-validation
detector = ConformalDetector(
    detector=LOF(),
    strategy=CrossValidation(k=5)
)

# Weighted conformal with JaB+
detector = ConformalDetector(
    detector=LOF(),
    strategy=JackknifeBootstrap(n_bootstraps=100),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
)
```

## References

- **Kim, B., Xu, C., & Barber, R. F. (2020)**. *Predictive Inference Is Free with the Jackknife+-after-Bootstrap.* Advances in Neural Information Processing Systems (NeurIPS), 33. [JaB+ method for bootstrap-based conformal calibration]

- **Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021)**. *Predictive Inference with the Jackknife+*. The Annals of Statistics, 49(1), 486-507. [Jackknife+ method with improved finite-sample efficiency]

- **Vovk, V., Gammerman, A., & Shafer, G. (2005)**. *Algorithmic Learning in a Random World*. Springer. [Foundational work on conformal prediction and cross-conformal prediction]

- **Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018)**. *Distribution-Free Predictive Inference for Regression*. Journal of the American Statistical Association, 113(523), 1094-1111. [Split conformal prediction with theoretical guarantees]

## Next Steps

- See [choosing strategies](choosing_strategies.md) for detailed decision framework
- Learn about [conformal inference](conformal_inference.md) for theoretical foundations
- Check [input validation](input_validation.md) for parameter constraints
