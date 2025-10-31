# Best Practices Guide

This guide provides recommendations for using nonconform effectively in different scenarios with the new API.

## Data Preparation

### 1. Data Quality

- Ensure your data is clean and preprocessed
- Handle missing values appropriately
- Normalize or standardize features when necessary
- Remove or handle outliers in the training data
- Check for data leakage between training and test sets

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Data cleaning pipeline
def prepare_data(X):
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    return X_scaled, imputer, scaler
```

### 2. Feature Engineering

- Use domain knowledge to create relevant features
- Consider feature selection to reduce dimensionality
- Handle categorical variables appropriately
- Create features that capture temporal patterns if applicable
- Consider feature interactions

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

def engineer_features(X, categorical_cols=None, k_best=None):
    """Feature engineering pipeline."""
    X_engineered = X.copy()

    # Handle categorical variables
    if categorical_cols:
        encoder = LabelEncoder()
        for col in categorical_cols:
            X_engineered[:, col] = encoder.fit_transform(X_engineered[:, col])

    # Feature selection
    if k_best:
        selector = SelectKBest(f_classif, k=k_best)
        X_engineered = selector.fit_transform(X_engineered, y_normal_indicator)

    return X_engineered
```

## Model Selection

### 1. Choosing a Detector

Consider the following when selecting a detector:

#### Data Size Considerations
- **Small datasets (< 1,000 samples)**: Use simpler models (IsolationForest, LOF)
- **Medium datasets (1,000-100,000)**: Most detectors work well
- **Large datasets (> 100,000)**: Consider scalable models, use parallel processing
- **High-dimensional data**: Use PCA-based preprocessing or specialized methods

#### Data Characteristics
- **Linear patterns**: Use PCA, OCSVM
- **Non-linear patterns**: Use IsolationForest, LOF, KNN
- **Complex patterns**: Use deep learning models when available
- **Temporal data**: Consider features that capture time dependencies

#### Computational Resources
```python
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN

# Fast detectors for large datasets
fast_detectors = {
    'IsolationForest': IForest(contamination=0.1, n_jobs=-1),
    'LOF': LOF(contamination=0.1, n_jobs=-1),
}

# Accurate but slower detectors
accurate_detectors = {
    'KNN': KNN(contamination=0.1),
    'OCSVM': OCSVM(contamination=0.1)
}
```

### 2. Ensemble Methods

Consider using multiple detectors for improved robustness:

```python
from nonconform.detection.standard import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func.enums import Aggregation
from scipy.stats import false_discovery_control

# Create multiple detectors
detectors = {
    'LOF': LOF(contamination=0.1),
    'IForest': IForest(contamination=0.1, n_jobs=-1),
    'OCSVM': OCSVM(contamination=0.1)
}

# Get p-values from each detector
all_p_values = {}
strategy = SplitStrategy(calibration_size=0.2)

for name, detector in detectors.items():
    conf_detector = ConformalDetector(
        detector=detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    conf_detector.fit(X_train)
    p_values = conf_detector.predict(X_test, raw=False)
    all_p_values[name] = p_values

# Combine results (simple approach: use minimum p-value)
ensemble_p_values = np.minimum.reduce(list(all_p_values.values()))
ensemble_discoveries = false_discovery_control(ensemble_p_values, method='bh') < 0.05
```

## Conformal Strategy Selection

### 1. Split Strategy

Best for:
- Large datasets (> 10,000 samples)
- When computational efficiency is important
- When you have enough data for reliable calibration

```python
from nonconform.strategy import Split

# For large datasets
strategy = Split(n_calib=0.2)  # Use 20% for calibration

# For very large datasets, use absolute number
strategy = Split(n_calib=1000)  # Use 1000 samples
```

### 2. Jackknife (Leave-One-Out)

Best for:
- Small datasets (< 1,000 samples)
- When you need maximum statistical power
- When computational cost is not a primary concern

```python
from nonconform.strategy import Jackknife

# For small datasets where every sample matters
strategy = Jackknife()
```

### 3. Bootstrap

Best for:
- Medium-sized datasets (1,000-10,000 samples)
- When you need robust estimates
- When you want to balance efficiency and power

```python
from nonconform.strategy import Bootstrap

# Balanced approach for medium datasets
strategy = Bootstrap(
    n_bootstraps=50,
    resampling_ratio=0.8
)
```

### 4. Cross-Validation

Best for:
- When you want to use all data efficiently
- Medium to large datasets
- When you need stable performance estimates

```python
from nonconform.strategy import CrossValidation

# Good balance of efficiency and stability
strategy = CrossValidation(k=5)
```

## Calibration Best Practices

### 1. Calibration Set Size

```python
def choose_calibration_strategy(n_samples):
    """Choose appropriate strategy based on dataset size."""
    if n_samples < 500:
        return Jackknife()
    elif n_samples < 2000:
        return Bootstrap(n_bootstraps=50, resampling_ratio=0.8)
    elif n_samples < 10000:
        return CrossValidation(k=5)
    else:
        # Use absolute number for very large datasets
        calib_size = min(2000, int(0.2 * n_samples))
        return Split(calib_size=calib_size)
```

### 2. Calibration Data Quality

- Ensure calibration data is representative of normal class
- Avoid using contaminated data for calibration
- Consider stratified sampling for balanced calibration

```python
def validate_calibration_data(X_train, contamination_rate=0.05):
    """Validate that calibration data is clean."""
    # Use a simple detector to identify potential anomalies in training data
    temp_detector = IForest(contamination=contamination_rate)
    temp_detector.fit(X_train)
    anomaly_scores = temp_detector.decision_function(X_train)

    # Keep only the most normal samples for calibration
    normal_threshold = np.percentile(anomaly_scores, (1 - contamination_rate) * 100)
    clean_indices = anomaly_scores >= normal_threshold

    return X_train[clean_indices]
```

## FDR Control Best Practices

### 1. Alpha Selection

```python
def choose_alpha_level(application_type):
    """Choose appropriate alpha level based on application."""
    alpha_levels = {
        'critical_safety': 0.001,      # Medical devices, safety systems
        'financial': 0.01,             # Fraud detection, trading
        'security': 0.01,              # Intrusion detection
        'quality_control': 0.05,       # Manufacturing, general QC
        'exploratory': 0.1,            # Research, data exploration
        'monitoring': 0.05             # System monitoring
    }
    return alpha_levels.get(application_type, 0.05)
```

### 2. Multiple Testing Scenarios

```python
from scipy.stats import false_discovery_control

def apply_fdr_control(p_values, alpha=0.05, method='bh'):
    """Apply FDR control with proper validation."""
    # Validate p-values
    if np.any(p_values < 0) or np.any(p_values > 1):
        raise ValueError("P-values must be between 0 and 1")

    # Apply FDR control
    adjusted_p_values = false_discovery_control(p_values, method=method, alpha=alpha)
    discoveries = adjusted_p_values < alpha

    print(f"Original detections: {(p_values < alpha).sum()}")
    print(f"FDR-controlled discoveries: {discoveries.sum()}")
    print(f"Reduction: {(p_values < alpha).sum() - discoveries.sum()}")

    return discoveries, adjusted_p_values
```

## Performance Monitoring

### 1. Key Metrics to Track

```python
def calculate_performance_metrics(y_true, discoveries):
    """Calculate comprehensive performance metrics."""
    if len(y_true) != len(discoveries):
        raise ValueError("y_true and discoveries must have same length")

    true_positives = np.sum(discoveries & (y_true == 1))
    false_positives = np.sum(discoveries & (y_true == 0))
    true_negatives = np.sum(~discoveries & (y_true == 0))
    false_negatives = np.sum(~discoveries & (y_true == 1))

    # Calculate metrics
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * precision * recall / max(1e-10, precision + recall)

    # FDR calculation
    fdr = false_positives / max(1, true_positives + false_positives)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fdr': fdr,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'discoveries': discoveries.sum()
    }
```

### 2. Performance Monitoring Pipeline

```python
import time
import psutil
import os

class PerformanceMonitor:
    """Monitor detector performance over time."""

    def __init__(self):
        self.metrics_history = []

    def monitor_prediction(self, detector, X_test, y_true=None):
        """Monitor a single prediction run."""
        # Time the prediction
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        p_values = detector.predict(X_test, raw=False)

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Apply FDR control
        discoveries, _ = apply_fdr_control(p_values)

        metrics = {
            'timestamp': time.time(),
            'prediction_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'n_samples': len(X_test),
            'discoveries': discoveries.sum(),
            'p_value_stats': {
                'min': p_values.min(),
                'max': p_values.max(),
                'mean': p_values.mean(),
                'std': p_values.std()
            }
        }

        # Add performance metrics if ground truth available
        if y_true is not None:
            perf_metrics = calculate_performance_metrics(y_true, discoveries)
            metrics.update(perf_metrics)

        self.metrics_history.append(metrics)
        return metrics
```

## Production Deployment

### 1. Model Updates and Drift Detection

```python
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

class ModelDriftDetector:
    """Detect when model needs updating due to drift."""

    def __init__(self, baseline_data, drift_threshold=0.05):
        self.baseline_data = baseline_data
        self.drift_threshold = drift_threshold

    def detect_drift(self, new_data):
        """Detect distribution drift using KS test."""
        drift_detected = False
        p_values = []

        for i in range(new_data.shape[1]):
            _, p_value = ks_2samp(
                self.baseline_data[:, i],
                new_data[:, i]
            )
            p_values.append(p_value)

            if p_value < self.drift_threshold:
                drift_detected = True

        return drift_detected, p_values
```

### 2. Scalable Batch Processing

```python
class ScalableAnomalyDetector:
    """Scalable anomaly detection for production."""

    def __init__(self, detector_config, batch_size=1000):
        self.detector_config = detector_config
        self.batch_size = batch_size
        self.detector = None

    def fit(self, X_train):
        """Fit detector on training data."""
        # Use appropriate strategy based on data size
        strategy = choose_calibration_strategy(len(X_train))

        self.detector = ConformalDetector(
            detector=self.detector_config['detector'],
            strategy=strategy,
            aggregation=self.detector_config['aggregation'],
            seed=self.detector_config['seed'],
            silent=self.detector_config.get('silent', True)
        )

        self.detector.fit(X_train)

    def predict_batch(self, X_test):
        """Predict on large datasets using batching."""
        import itertools

        all_p_values = []

        for batch in itertools.batched(X_test, self.batch_size):
            batch_p_values = self.detector.predict(batch, raw=False)
            all_p_values.extend(batch_p_values)

        return np.array(all_p_values)
```

## Code Organization

### 1. Configuration Management

```python
from dataclasses import dataclass
from typing import Optional
from nonconform.utils.func.enums import Aggregation


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection pipeline."""
    alpha: float = 0.05
    calibration_size: float = 0.2  # Can be float (ratio) or int (absolute)
    detector_type: str = "iforest"
    aggregation: Aggregation = Aggregation.MEDIAN
    seed: int = 42
    silent: bool = True
    batch_size: int = 1000
    fdr_method: str = 'bh'

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")

        if isinstance(self.calibration_size, float) and not 0 < self.calibration_size < 1:
            raise ValueError("Calibration size ratio must be between 0 and 1")
```

### 2. Complete Pipeline Implementation

```python
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

class AnomalyDetectionPipeline:
    """Complete anomaly detection pipeline."""

    DETECTOR_MAP = {
        'iforest': IForest,
        'lof': LOF,
        'ocsvm': OCSVM
    }

    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.detector = None
        self.performance_monitor = PerformanceMonitor()
        self.drift_detector = None

    def _create_detector(self):
        """Create base detector from configuration."""
        detector_class = self.DETECTOR_MAP[self.config.detector_type]
        return detector_class(contamination=0.1)

    def _create_strategy(self, n_samples):
        """Create strategy based on dataset size."""
        return choose_calibration_strategy(n_samples)

    def fit(self, X_train):
        """Fit the complete pipeline."""
        # Validate and prepare data
        X_clean = validate_calibration_data(X_train)

        # Create components
        base_detector = self._create_detector()
        strategy = self._create_strategy(len(X_clean))

        # Create conformal detector
        self.detector = ConformalDetector(
            detector=base_detector,
            strategy=strategy,
            aggregation=self.config.aggregation,
            seed=self.config.seed,
            silent=self.config.silent
        )

        # Fit detector
        self.detector.fit(X_clean)

        # Initialize drift detector
        self.drift_detector = ModelDriftDetector(X_clean)

        print(f"Pipeline fitted with {len(X_clean)} samples")
        print(f"Strategy: {type(strategy).__name__}")
        print(f"Calibration set size: {len(self.detector.calibration_set)}")

    def predict(self, X_test, y_true=None, check_drift=True):
        """Make predictions with full monitoring."""
        if self.detector is None:
            raise ValueError("Pipeline must be fitted before prediction")

        # Check for drift
        if check_drift and self.drift_detector:
            drift_detected, _ = self.drift_detector.detect_drift(X_test)
            if drift_detected:
                print("WARNING: Distribution drift detected!")

        # Make predictions
        if len(X_test) > self.config.batch_size:
            # Use batch processing for large datasets
            p_values = self._predict_batch(X_test)
        else:
            p_values = self.detector.predict(X_test, raw=False)

        # Apply FDR control
        discoveries, adjusted_p_values = apply_fdr_control(
            p_values,
            alpha=self.config.alpha,
            method=self.config.fdr_method
        )

        # Monitor performance
        metrics = self.performance_monitor.monitor_prediction(
            self.detector, X_test, y_true
        )

        return {
            'discoveries': discoveries,
            'p_values': p_values,
            'adjusted_p_values': adjusted_p_values,
            'metrics': metrics
        }

    def _predict_batch(self, X_test):
        """Batch prediction for large datasets."""
        scalable_detector = ScalableAnomalyDetector(
            {
                'detector': self._create_detector(),
                'aggregation': self.config.aggregation,
                'seed': self.config.seed,
                'silent': self.config.silent
            },
            batch_size=self.config.batch_size
        )

        # Note: In practice, you'd want to reuse the fitted detector
        # This is simplified for demonstration
        return scalable_detector.predict_batch(X_test)
```

### 3. Usage Example

```python
# Configuration
config = AnomalyDetectionConfig(
    alpha=0.05,
    calibration_size=0.2,
    detector_type="iforest",
    aggregation=Aggregation.MEDIAN,
    fdr_method='bh'
)

# Create and use pipeline
pipeline = AnomalyDetectionPipeline(config)
pipeline.fit(X_train)

# Make predictions
results = pipeline.predict(X_test, y_true=y_test)

print(f"Discoveries: {results['discoveries'].sum()}")
print(f"Performance metrics: {results['metrics']}")
```

This comprehensive approach ensures robust, scalable, and maintainable anomaly detection systems using the new nonconform API.
