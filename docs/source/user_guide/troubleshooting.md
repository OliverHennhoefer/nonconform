# Troubleshooting Guide

This guide addresses common issues you might encounter while using nonconform and provides solutions.

## Common Issues and Solutions

### 1. ImportError: Cannot import DetectorConfig

**Problem**: Getting import errors when trying to use DetectorConfig.

**Solution**: DetectorConfig has been removed. Use direct parameters instead:

```python
# Old API (deprecated)
from nonconform.detection.configuration import DetectorConfig

detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    config=DetectorConfig(alpha=0.1)
)

# New API
from nonconform.detection.standard import ConformalDetector
from nonconform.strategy import SplitStrategy
from nonconform.utils.func.enums import Aggregation

detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

### 2. AttributeError: predict() has no parameter 'output'

**Problem**: Using the old output parameter in predict() method.

**Solution**: Replace `output` with `raw` parameter:

```python
# Old API (deprecated)
p_values = detector.predict(X, output="p-value")
scores = detector.predict(X, output="score")

# New API
p_values = detector.predict(X, raw=False)  # Get p-values
scores = detector.predict(X, raw=True)     # Get raw scores
```

### 3. Memory Issues

**Problem**: Running out of memory when using large datasets or certain detectors.

**Solutions**:
- Use batch processing for large datasets
- Consider using more memory-efficient detectors (e.g., IsolationForest instead of KNN)
- Reduce the calibration set size
- Use sparse data structures when possible

```python
import itertools

def process_in_batches(detector, X, batch_size=1000):
    """Process large datasets in batches."""
    results = []
    for batch in itertools.batched(X, batch_size):
        batch_results = detector.predict(batch, raw=False)
        results.extend(batch_results)
    return np.array(results)
```

### 4. Slow Performance

**Problem**: Processing takes too long, especially with large datasets.

**Solutions**:
- Use faster detectors (e.g., IsolationForest, LOF)
- Reduce the calibration set size
- Use batch processing
- Configure logging to hide progress bars: `logging.getLogger('nonconform').setLevel(logging.WARNING)`
- Profile your code to identify bottlenecks

```python
import time

# Time your detector
start_time = time.time()
detector.fit(X_train)
fit_time = time.time() - start_time

start_time = time.time()
p_values = detector.predict(X_test, raw=False)
predict_time = time.time() - start_time

print(f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
```

### 5. Invalid P-values

**Problem**: P-values don't seem to be properly calibrated or all values are extreme.

**Solutions**:
- Ensure your calibration data is representative of the normal class
- Check for data leakage between training and calibration sets
- Verify that the detector is properly fitted
- Consider using a different conformal strategy
- Check for violations of the exchangeability assumption

```python
def validate_p_values(p_values):
    """Validate p-value distribution."""
    print(f"P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")
    print(f"P-value mean: {p_values.mean():.4f}")
    print(f"P-value std: {p_values.std():.4f}")

    # Check for uniform distribution (expected under null hypothesis)
    from scipy.stats import kstest
    ks_stat, ks_p = kstest(p_values, 'uniform')
    print(f"KS test for uniformity: stat={ks_stat:.4f}, p={ks_p:.4f}")

    if ks_p < 0.05:
        print("WARNING: P-values may not be well-calibrated")
```

### 6. High False Discovery Rate

**Problem**: Too many false positives even with FDR control.

**Solutions**:
- Increase the calibration set size
- Use a more conservative α level for FDR control
- Consider using weighted conformal p-values if there's covariate shift
- Try different detectors
- Check for data quality issues

```python
from scipy.stats import false_discovery_control

# Use more conservative FDR control
adjusted_p_values = false_discovery_control(p_values, method='by', alpha=0.01)  # More conservative
discoveries = adjusted_p_values < 0.01

# Monitor empirical FDR if ground truth is available
if y_true is not None:
    false_positives = np.sum(discoveries & (y_true == 0))
    empirical_fdr = false_positives / max(1, discoveries.sum())
    print(f"Empirical FDR: {empirical_fdr:.3f}")
```

### 7. Low Detection Power

**Problem**: Missing too many anomalies.

**Solutions**:
- Use less conservative α levels
- Use more powerful detectors
- Consider using ensemble methods
- Try different conformal strategies (e.g., bootstrap, cross-validation)
- Check if the anomalies are well-separated from normal data

```python
# Try multiple strategies for comparison
from nonconform.strategy import Bootstrap
from nonconform.strategy import CrossValidation

strategies = {
    'Split': Split(calib_size=0.2),
    'Bootstrap': Bootstrap(n_bootstraps=100, resampling_ratio=0.8),
    'CV': CrossValidation(k=5)
}

for name, strategy in strategies.items():
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    detector.fit(X_train)
    p_vals = detector.predict(X_test, raw=False)
    detections = (p_vals < 0.05).sum()
    print(f"{name}: {detections} detections")
```

### 8. Strategy Import Issues

**Problem**: Cannot import strategy classes with old import paths.

**Solution**: Update import statements to use new module structure:

```python
# Old imports (deprecated)
# Removed - use individual imports

# New imports
from nonconform.strategy import Split
from nonconform.strategy import CrossValidation
from nonconform.strategy import Jackknife
from nonconform.strategy import Bootstrap
```

### 9. Parameter Name Changes

**Problem**: Using old parameter names that have been renamed.

**Solution**: Update parameter names:

```python
# Old parameter names
Split(calibration_size=0.2)           # -> calib_size=0.2
CrossValidation(n_splits=5)            # -> k=5
Bootstrap(sample_ratio=0.8) # -> resampling_ratio=0.8
```

### 10. Integration Issues

**Problem**: Problems integrating with other libraries or custom detectors.

**Solutions**:
- Ensure your detector implements the PyOD BaseDetector interface
- Check for version compatibility
- Verify that the detector's output format matches expectations
- Use the correct aggregation enum values

```python
from nonconform.utils.func.enums import Aggregation

# Correct usage of aggregation enums
detector = ConformalDetector(
    detector=custom_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,  # Not "median"
    seed=42
)
```

## Debugging Tips

### 1. Enable Verbose Mode

```python
import logging

# Enable progress bars and detailed output
logging.getLogger('nonconform').setLevel(logging.INFO)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# For debugging, use DEBUG level for maximum verbosity
logging.getLogger('nonconform').setLevel(logging.DEBUG)
```

### 2. Check Intermediate Results

```python
# Get raw scores before p-value conversion
raw_scores = detector.predict(X_test, raw=True)
p_values = detector.predict(X_test, raw=False)

print(f"Raw scores range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
print(f"P-values range: [{p_values.min():.4f}, {p_values.max():.4f}]")

# Check calibration set
print(f"Calibration set size: {len(detector.calibration_set)}")
print(f"Calibration scores range: [{min(detector.calibration_set):.4f}, {max(detector.calibration_set):.4f}]")
```

### 3. Validate Data

```python
def validate_input_data(X):
    """Validate input data for common issues."""
    print("=== Data Validation ===")
    print(f"Shape: {X.shape}")
    print(f"Data type: {X.dtype}")

    # Check for NaN values
    nan_count = np.isnan(X).sum()
    print(f"NaN values: {nan_count}")

    # Check for infinite values
    inf_count = np.isinf(X).sum()
    print(f"Infinite values: {inf_count}")

    # Check data ranges
    print(f"Data range: [{X.min():.4f}, {X.max():.4f}]")

    # Check for constant features
    constant_features = np.sum(X.std(axis=0) == 0)
    print(f"Constant features: {constant_features}")

    if nan_count > 0 or inf_count > 0:
        print("WARNING: Data contains NaN or infinite values")

    if constant_features > 0:
        print("WARNING: Data contains constant features")

# Example usage
validate_input_data(X_train)
validate_input_data(X_test)
```

### 4. Monitor Memory Usage

```python
import psutil
import os

def print_memory_usage(label=""):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage {label}: {memory_mb:.2f} MB")

# Monitor memory during processing
print_memory_usage("before fitting")
detector.fit(X_train)
print_memory_usage("after fitting")
p_values = detector.predict(X_test, raw=False)
print_memory_usage("after prediction")
```

### 5. Debug Weighted Conformal Issues

```python
def debug_weighted_conformal(detector, X_train, X_test):
    """Debug weighted conformal detection specifically."""
    print("=== Weighted Conformal Debug ===")

    # Check if it's actually a weighted detector
    from nonconform.detection.weighted import ConformalDetector
    if not isinstance(detector, ConformalDetector):
        print("WARNING: Not a ConformalDetector")
        return

    # Fit and check calibration samples
    detector.fit(X_train)

    if hasattr(detector, 'calibration_samples'):
        print(f"Calibration samples stored: {len(detector.calibration_samples)}")
        if len(detector.calibration_samples) == 0:
            print("ERROR: No calibration samples stored")

    # Check for distribution shift
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])

    clf = LogisticRegression(random_state=42)
    scores = cross_val_score(clf, X_combined, y_combined, cv=5)
    shift_score = scores.mean()

    print(f"Distribution shift score: {shift_score:.3f}")
    if shift_score > 0.7:
        print("Significant shift detected - weighted conformal recommended")
    elif shift_score < 0.6:
        print("Minimal shift - standard conformal may suffice")
```

## Performance Optimization

### 1. Batch Processing

```python
import itertools

def optimized_batch_processing(detector, X, batch_size=1000):
    """Optimized batch processing for large datasets."""
    n_samples = len(X)
    results = np.empty(n_samples)

    start_idx = 0
    for i, batch in enumerate(itertools.batched(X, batch_size)):
        batch_results = detector.predict(batch, raw=False)
        end_idx = start_idx + len(batch)
        results[start_idx:end_idx] = batch_results
        start_idx = end_idx

        if i % 10 == 0:  # Progress update
            print(f"Processed {i + 1} batches")

    return results
```

### 2. Strategy-Specific Optimizations

```python
# For large datasets, use split strategy with smaller calibration
strategy = SplitStrategy(calibration_size=0.1)  # Smaller calibration set

# For small datasets, use bootstrap for stability
strategy = BootstrapStrategy(n_bootstraps=50, sample_ratio=0.8)

# For medium datasets, use cross-validation
strategy = CrossValidationStrategy(n_splits=5)
```

### 3. Detector Selection for Performance

```python
# Fast detectors for large datasets
fast_detectors = [
    IsolationForest(contamination=0.1, n_jobs=-1),  # Parallel processing
    LOF(contamination=0.1, n_jobs=-1),
    OCSVM(contamination=0.1)
]

# Avoid expensive detectors for large datasets
# - KNN with large k
# - Complex neural networks
# - High-dimensional methods without dimensionality reduction
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the New API**: Ensure you're using the updated API with direct parameters instead of DetectorConfig
2. **Update Import Statements**: Use the new module structure for strategy imports
3. **Verify Parameter Names**: Check that parameter names match the new API
4. **Check the [GitHub Issues](https://github.com/OliverHennhoefer/nonconform/issues)** for similar problems
5. **Search the [Discussions](https://github.com/OliverHennhoefer/nonconform/discussions)** for solutions
6. **Create a new issue** with:
   - A minimal reproducible example using the new API
   - Expected vs actual behavior
   - System information (Python version, nonconform version, etc.)
   - Relevant error messages
   - Whether you're migrating from the old API

## Migration Checklist

When migrating from older versions of nonconform:

- [ ] Remove `DetectorConfig` imports and usage
- [ ] Update detector initialization to use direct parameters
- [ ] Change `output="p-value"` to `raw=False`
- [ ] Change `output="score"` to `raw=True`
- [ ] Update strategy imports to use new module structure
- [ ] Replace old parameter names with new ones
- [ ] Add FDR control using `scipy.stats.false_discovery_control`
- [ ] Test with small datasets first
- [ ] Update any custom code that depends on the old API
- [ ] Replace `silent=True/False` with logging configuration

## Logging Configuration

nonconform uses Python's standard logging framework to control progress bars and informational output. This provides more flexibility than the old `silent` parameter.

### Basic Logging Setup

```python
import logging

# Show progress bars and info messages (default for development)
logging.getLogger('nonconform').setLevel(logging.INFO)

# Hide progress bars (recommended for production)
logging.getLogger('nonconform').setLevel(logging.WARNING)

# Show everything including debug info
logging.getLogger('nonconform').setLevel(logging.DEBUG)
```

### Common Logging Scenarios

**Production Environment:**
```python
import logging

# Hide all progress bars and info messages
logging.getLogger('nonconform').setLevel(logging.WARNING)

detector = ConformalDetector(detector=IForest(), strategy=Split())
detector.fit(X_train)  # No progress output
```

**Development Environment:**
```python
import logging

# Show progress bars for monitoring
logging.basicConfig(level=logging.INFO)

detector = ConformalDetector(detector=IForest(), strategy=CrossValidation(k=5))
detector.fit(X_train)  # Shows CV fold progress
```

**Selective Logging:**
```python
import logging

# Show progress but hide specific module info
logging.getLogger('nonconform').setLevel(logging.INFO)
logging.getLogger('nonconform.strategy.bootstrap').setLevel(logging.WARNING)

# This will show aggregation progress but hide bootstrap configuration details
```

**Debug Mode:**
```python
import logging

# Maximum verbosity for troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# This will show all internal operations and warnings
detector = ConformalDetector(detector=LOF(), strategy=Bootstrap())
detector.fit(X_train)
```

### Logger Hierarchy

nonconform uses the following logger hierarchy:
- `nonconform`: Root logger for all nonconform output
- `nonconform.detection.*`: Detector-specific logging
- `nonconform.strategy.*`: Strategy-specific logging
- `nonconform.utils.*`: Utility function logging

You can configure specific loggers for fine-grained control over output.
