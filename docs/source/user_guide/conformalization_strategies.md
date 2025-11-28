# Conformalization Strategies

Calibration strategies for conformal anomaly detection with different trade-offs between computational efficiency and statistical robustness.

## Available Strategies

### Split Strategy

Simple train/calibration split for fast, straightforward conformal prediction.

```python
from nonconform.strategy import Split

# Use 30% of data for calibration
strategy = Split(n_calib=0.3)

# Use fixed number of samples for calibration
strategy = Split(n_calib=100)
```

**Characteristics:**
- **Fastest** computation
- **Simplest** implementation
- **Least robust** for small datasets
- **Memory efficient**

### Cross-Validation Strategy

K-fold cross-validation for robust calibration using all data.

```python
from nonconform.strategy import CrossValidation

# 5-fold cross-validation
strategy = CrossValidation(n_folds=5, plus=False)

# Enable plus mode for tighter prediction intervals
strategy = CrossValidation(n_folds=5, plus=True)
```

**Characteristics:**
- **Most robust** calibration
- **Uses all data** for both training and calibration
- **Higher computational cost**
- **Recommended for small datasets**

### Bootstrap Strategy

Bootstrap resampling with configurable ensemble parameters.

```python
from nonconform.strategy import Bootstrap

# Basic bootstrap with 100 models
strategy = Bootstrap(n_bootstraps=100, resampling_ratio=0.8)

# Automated configuration
strategy = Bootstrap(n_calib=200)  # Auto-calculate other parameters
```

**Characteristics:**
- **Flexible ensemble** size
- **Uncertainty quantification**
- **Robust to outliers**
- **Configurable computational cost**

### Jackknife Strategy

Leave-one-out cross-validation for maximum data utilization [[Barber et al., 2021](#references)].

```python
from nonconform.strategy import Jackknife

# Standard jackknife
strategy = Jackknife(plus=False)

# Jackknife+ for tighter intervals
strategy = Jackknife(plus=True)
```

**Characteristics:**
- **Maximum data utilization**
- **Computationally intensive**
- **Best for very small datasets**
- **Provides individual sample influence**

## Strategy Selection Guide

| Dataset Size | Computational Budget | Recommendation |
|-------------|---------------------|----------------|
| Large (>1000) | Low | Split |
| Large (>1000) | High | CrossValidation |
| Medium (100-1000) | Any | CrossValidation |
| Small (<100) | Any | Jackknife |

## Plus Mode

All strategies support "plus" mode for tighter prediction intervals [[Barber et al., 2021](#references)]:

```python
# Enable plus mode for any strategy
strategy = CrossValidation(n_folds=5, plus=True)
strategy = Bootstrap(n_bootstraps=50, plus=True)
strategy = Jackknife(plus=True)
```

**Plus mode provides:**
- Higher statistical efficiency in theory [[Barber et al., 2021](#references)]
- Better finite-sample properties
- Slightly higher computational cost

The "plus" suffix (e.g., Jackknife+, CV+) indicates a refined version that often produces shorter prediction intervals while maintaining coverage guarantees.

## Performance Comparison

| Strategy | Training Time | Memory Usage | Calibration Quality |
|----------|---------------|--------------|-------------------|
| Split | Fast | Low | Good |
| CrossValidation | Medium | Medium | Excellent |
| Bootstrap | Medium-High | Medium-High | Very Good |
| Jackknife | Slow | High | Excellent |

## Integration with Detectors

All strategies work with any conformal detector:

```python
from nonconform.detection import ConformalDetector
from pyod.models.lof import LOF

# Standard conformal with cross-validation
detector = ConformalDetector(
    detector=LOF(),
    strategy=CrossValidation(k=5)
)

# Weighted conformal with bootstrap
from nonconform.detection.weight import LogisticWeightEstimator

detector = ConformalDetector(
    detector=LOF(),
    strategy=Bootstrap(n_bootstraps=100),
    weight_estimator=LogisticWeightEstimator(seed=42)
)
```

## References

- **Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021)**. *Predictive Inference with the Jackknife+*. The Annals of Statistics, 49(1), 486-507. [Jackknife+ method with improved finite-sample efficiency]

- **Vovk, V., Gammerman, A., & Shafer, G. (2005)**. *Algorithmic Learning in a Random World*. Springer. [Foundational work on conformal prediction and cross-conformal prediction]

- **Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018)**. *Distribution-Free Predictive Inference for Regression*. Journal of the American Statistical Association, 113(523), 1094-1111. [Split conformal prediction with theoretical guarantees]

## Next Steps

- See [choosing strategies](choosing_strategies.md) for detailed decision framework
- Learn about [conformal inference](conformal_inference.md) for theoretical foundations
- Check [input validation](input_validation.md) for parameter constraints
