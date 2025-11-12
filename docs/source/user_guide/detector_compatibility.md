# PyOD Detector Compatibility Guide

This guide explains which PyOD detectors work with nonconform, why some are restricted, and how to choose the right detector for your anomaly detection task.

## Compatibility Overview

Nonconform is designed to work with PyOD detectors that support **one-class classification** - training on normal data only without knowing anomaly labels. This is essential for conformal prediction's theoretical guarantees.

### ✅ Compatible Detectors

All detectors listed below are fully compatible and tested with nonconform:

| Detector | Class | Best For | Performance |
|----------|-------|----------|-------------|
| **Isolation Forest** | `IForest` | High-dimensional data, large datasets | Fast |
| **Local Outlier Factor** | `LOF` | Dense clusters, local anomalies | Medium |
| **K-Nearest Neighbors** | `KNN` | Simple distance-based detection | Fast |
| **One-Class SVM** | `OCSVM` | Complex boundaries, small datasets | Slow |
| **Principal Component Analysis** | `PCA` | Linear anomalies, interpretability | Fast |
| **Empirical Cumulative Distribution** | `ECOD` | Parameter-free, robust | Fast |
| **Copula-Based Detection** | `COPOD` | Correlation-based anomalies | Medium |
| **Histogram-Based Outlier** | `HBOS` | Feature independence assumptions | Fast |
| **Gaussian Mixture Model** | `GMM` | Probabilistic modeling | Medium |
| **Auto-Encoder** | `AutoEncoder` | Deep learning, complex patterns | Slow |
| **Variational Auto-Encoder** | `VAE` | Probabilistic deep learning | Slow |

### ❌ Restricted Detectors

These detectors are **forbidden** and will raise `ValueError` when used:

| Detector | Class | Reason for Restriction |
|----------|-------|----------------------|
| **Cluster-Based LOF** | `CBLOF` | Requires clustering labels during training |
| **Connectivity-Based Outlier** | `COF` | Needs connectivity information not available in one-class setting |
| **R-Graph** | `RGraph` | Requires graph structure incompatible with one-class training |
| **Sampling-Based** | `Sampling` | Needs sampling strategies requiring anomaly examples |
| **Stochastic Outlier Selection** | `SOS` | Requires pre-computed outlier probabilities |

## One-Class Training Requirements

### Why One-Class Matters

Conformal prediction requires training detectors on **calibration data** that follows the same distribution as test data. In anomaly detection:

1. **Training phase:** Uses only normal samples (no anomaly labels)
2. **Calibration phase:** Computes scores on held-out normal samples
3. **Prediction phase:** Converts new sample scores to valid p-values

Detectors requiring anomaly examples during training violate this assumption and cannot provide valid conformal guarantees.

### Automatic Configuration

Nonconform automatically configures compatible detectors for one-class training:

```python
# Before conformal wrapping
detector = IForest(contamination=0.1)  # Expects 10% anomalies

# After automatic configuration
# contamination → sys.float_info.min (essentially 0)
# n_jobs → -1 (use all cores)
# random_state → seed (for reproducibility)
```

## Detector Selection Guide

### By Data Characteristics

**High-dimensional data (>100 features):**
- **Primary choice:** Isolation Forest
- **Alternative:** Auto-Encoder (if computational budget allows)
- **Avoid:** LOF (curse of dimensionality)

**Low-dimensional data (<20 features):**
- **Primary choice:** LOF or OCSVM
- **Alternative:** KNN or HBOS
- **Consider:** PCA for linear patterns

**Mixed data types (numerical + categorical):**
- **Primary choice:** HBOS (handles mixed types well)
- **Alternative:** COPOD
- **Avoid:** PCA (requires numerical data)

**Time-series data:**
- **Primary choice:** ECOD (parameter-free)
- **Alternative:** Isolation Forest
- **Consider:** Auto-Encoder for temporal patterns

### By Dataset Size

**Large datasets (>50,000 samples):**
- **Primary choice:** Isolation Forest (scales well)
- **Alternative:** ECOD (parameter-free scaling)
- **Avoid:** OCSVM (quadratic complexity)

**Medium datasets (1,000-50,000 samples):**
- **Primary choice:** LOF or KNN
- **Alternative:** Any detector based on requirements
- **Consider:** Ensemble approaches

**Small datasets (<1,000 samples):**
- **Primary choice:** OCSVM (good with limited data)
- **Alternative:** PCA (simple, interpretable)
- **Avoid:** Deep learning methods (insufficient data)

### By Performance Requirements

**Real-time inference (<10ms per sample):**
```python
# Fast detectors with minimal configuration
detectors = [
    PCA(n_components=10),
    HBOS(n_bins=10),
    KNN(n_neighbors=5, method='mean')
]
```

**Batch processing (seconds acceptable):**
```python
# Balanced accuracy and speed
detectors = [
    IForest(n_estimators=100),
    LOF(n_neighbors=20),
    ECOD()
]
```

**Offline analysis (minutes acceptable):**
```python
# Maximum accuracy configurations
detectors = [
    OCSVM(gamma='auto', nu=0.05),
    AutoEncoder(epoch_num=200, hidden_neuron_list=[128, 64, 32, 64, 128]),
    VAE(epoch_num=100, latent_dim=20)
]
```

## Detector-Specific Recommendations

### Isolation Forest (`IForest`)

**Best configuration:**
```python
from pyod.models.iforest import IForest

detector = IForest(
    behaviour="new",        # Use scikit-learn 0.22+ behavior
    n_estimators=100,       # Balance accuracy and speed
    max_samples="auto",     # Automatic subsampling
    random_state=42         # Will be set by nonconform
)
```

**Tuning tips:**
- Increase `n_estimators` for better accuracy (diminishing returns after 200)
- Use `max_samples=256` for very large datasets to control memory
- `max_features=1.0` for high-dimensional sparse data

### Local Outlier Factor (`LOF`)

**Best configuration:**
```python
from pyod.models.lof import LOF

detector = LOF(
    n_neighbors=20,         # Start with 20, tune based on data density
    algorithm='auto',       # Let sklearn choose optimal algorithm
    metric='minkowski'      # Standard Euclidean distance
)
```

**Tuning tips:**
- Increase `n_neighbors` for smoother decision boundaries
- Use `n_neighbors=min(50, n_samples//20)` as a rule of thumb
- Consider `metric='manhattan'` for high-dimensional data

### One-Class SVM (`OCSVM`)

**Best configuration:**
```python
from pyod.models.ocsvm import OCSVM

detector = OCSVM(
    kernel='rbf',           # Radial basis function kernel
    gamma='auto',           # Automatic gamma selection
    nu=0.05,               # Expected outlier fraction (keep low)
    shrinking=True         # Enable shrinking heuristic
)
```

**Tuning tips:**
- Start with `gamma='auto'`, then try `gamma='scale'`
- Keep `nu` small (0.01-0.1) for conformal prediction
- Use `kernel='linear'` for high-dimensional linear data

### ECOD (`ECOD`)

**Best configuration:**
```python
from pyod.models.ecod import ECOD

detector = ECOD()  # Parameter-free!
```

**Advantages:**
- No hyperparameter tuning required
- Robust across different data types
- Good baseline performance
- Fast and memory efficient

## Common Configuration Mistakes

### ❌ Wrong Contamination Values

```python
# DON'T: High contamination in training
detector = IForest(contamination=0.1)  # Assumes 10% anomalies

# DO: Let nonconform handle contamination
detector = IForest()  # Will be set to minimal value automatically
```

### ❌ Inappropriate Hyperparameters

```python
# DON'T: Too many neighbors for small datasets
detector = LOF(n_neighbors=100)  # On 500-sample dataset

# DO: Scale neighbors with dataset size
detector = LOF(n_neighbors=min(20, n_samples//10))
```

### ❌ Resource-Intensive Settings

```python
# DON'T: Memory-intensive settings for large data
detector = OCSVM(gamma=0.001)  # Very wide RBF kernel

# DO: Use automatic parameter selection
detector = OCSVM(gamma='auto')
```

## Testing Detector Compatibility

### Basic Compatibility Check

```python
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split


def test_detector_compatibility(detector, X_train, X_test):
    """Test if a detector works with nonconform."""
    try:
        conformal_detector = ConformalDetector(
            detector=detector,
            strategy=Split(n_calib=0.2),
            seed=42
        )
        conformal_detector.fit(X_train)
        p_values = conformal_detector.predict(X_test[:10])

        # Check if p-values are valid
        assert all(0 <= p <= 1 for p in p_values), "Invalid p-values"
        print(f"✓ {detector.__class__.__name__} is compatible")
        return True

    except Exception as e:
        print(f"✗ {detector.__class__.__name__} failed: {e}")
        return False
```

### Performance Benchmarking

```python
import time
from nonconform.utils.stat import false_discovery_rate, statistical_power

def benchmark_detector(detector, X_train, X_test, y_test):
    """Benchmark detector performance with conformal prediction."""
    conformal_detector = ConformalDetector(
        detector=detector,
        strategy=Split(n_calib=0.2),
        seed=42
    )

    # Measure fitting time
    start = time.time()
    conformal_detector.fit(X_train)
    fit_time = time.time() - start

    # Measure prediction time
    start = time.time()
    p_values = conformal_detector.predict(X_test)
    pred_time = time.time() - start

    # Calculate accuracy metrics
    from scipy.stats import false_discovery_control
    decisions = false_discovery_control(p_values, method='bh') <= 0.1
    fdr = false_discovery_rate(y_test, decisions)
    power = statistical_power(y_test, decisions)

    return {
        'detector': detector.__class__.__name__,
        'fit_time': fit_time,
        'pred_time': pred_time,
        'fdr': fdr,
        'power': power,
        'calibration_size': len(conformal_detector.calibration_set)
    }
```

## Best Practices Summary

1. **Start simple:** Begin with Isolation Forest and Split strategy for initial prototyping
2. **Validate compatibility:** Test any new detector with the compatibility check above
3. **Benchmark systematically:** Use the benchmarking function to compare options
4. **Consider your constraints:** Balance accuracy needs with computational resources
5. **Monitor in production:** Track FDR and power metrics to ensure continued performance
6. **Document choices:** Record which detector and strategy work best for your specific use case

## Getting Help

If you encounter issues with detector compatibility:

1. Check if the detector is in the forbidden list
2. Verify that your detector supports one-class training
3. Test with the compatibility check function above
4. Review PyOD documentation for detector-specific requirements
5. Consider alternative detectors with similar characteristics
