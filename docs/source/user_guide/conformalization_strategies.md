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

Leave-one-out cross-validation for maximum data utilization.

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

All strategies support "plus" mode for tighter prediction intervals:

```python
# Enable plus mode for any strategy
strategy = CrossValidation(n_folds=5, plus=True)
strategy = Bootstrap(n_bootstraps=50, plus=True)
strategy = Jackknife(plus=True)
```

**Plus mode provides:**
- Higher statistical efficiency in theory
- Better finite-sample properties
- Slightly higher computational cost

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
from nonconform.detection.standard import ConformalDetector
from nonconform.detection.weighted import ConformalDetector
from pyod.models.lof import LOF

# Standard conformal with cross-validation
detector = ConformalDetector(
    detector=LOF(),
    strategy=CrossValidation(k=5)
)

# Weighted conformal with bootstrap
detector = ConformalDetector(
    detector=LOF(),
    strategy=Bootstrap(n_bootstraps=100)
)
```
