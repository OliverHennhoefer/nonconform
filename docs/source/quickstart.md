# Quick Start

Get started with nonconform in minutes.

## What You'll Learn

By the end of this guide, you'll know how to:

1. Wrap any anomaly detector to get statistically valid p-values
2. Use FDR control to make principled anomaly detection decisions
3. Choose between different calibration strategies
4. Handle distribution shift with weighted conformal methods

**Prerequisites**: Familiarity with Python and basic anomaly detection concepts. No prior knowledge of conformal prediction required.

---

## Your First Conformal Detector

Let's start with a simple example that shows the core workflow:

```python
import numpy as np
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split

# Step 1: Load benchmark data
from oddball import Dataset, load

# Training data contains only "normal" observations
# Test data contains both normal and anomalous observations with ground truth labels
X_train, X_test, y_true = load(Dataset.SHUTTLE, setup=True, seed=42)

# Step 2: Wrap your detector with nonconform
detector = ConformalDetector(
    detector=IForest(random_state=42),  # Any anomaly detector
    strategy=Split(n_calib=0.3),         # Use 30% of training data for calibration
    seed=42
)

# Step 3: Fit on normal data
detector.fit(X_train)

# Step 4: Get p-values for test observations
# Low p-values indicate likely anomalies
p_values = detector.predict(X_test)

# Step 5: Apply FDR control to make decisions
# This ensures at most 5% of your "discoveries" are false positives
decisions = false_discovery_control(p_values, method='bh') < 0.05

print(f"Flagged {decisions.sum()} anomalies out of {len(X_test)} observations")
```

**What just happened?**

1. We trained an Isolation Forest on normal data
2. nonconform split some of that data for **calibration**—computing reference scores to compare against
3. For each test observation, nonconform computed a **p-value**: the probability of seeing a score this extreme if the observation were normal
4. FDR control adjusted these p-values to account for multiple testing, ensuring our false positive rate is controlled

---

## Loading Benchmark Datasets

For experimentation, use the `oddball` package which provides standard anomaly detection benchmarks:

```bash
pip install "nonconform[data]"
```

```python
from oddball import Dataset, load

# Load a dataset - automatically downloads and caches
x_train, x_test, y_test = load(Dataset.BREASTW, setup=True)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Anomaly rate: {y_test.mean():.1%}")
```

!!! info "Available Datasets"
    Common benchmarks include `BREASTW`, `SHUTTLE`, `THYROID`, `IONOSPHERE`, `MAMMOGRAPHY`, and more. See `oddball.list_available()` for the full list.

---

## Evaluating Results

nonconform provides metrics to evaluate your anomaly detection performance:

```python
from scipy.stats import false_discovery_control

from nonconform import (
    Aggregation,
    ConformalDetector,
    Split,
    false_discovery_rate,
    statistical_power,
)
from oddball import Dataset, load
from pyod.models.iforest import IForest

# Load data
x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

# Create and fit detector
detector = ConformalDetector(
    detector=IForest(random_state=42),
    strategy=Split(n_calib=0.3),
    aggregation=Aggregation.MEDIAN,
    seed=42
)
detector.fit(x_train)

# Get p-values and apply FDR control
p_values = detector.predict(x_test)
decisions = false_discovery_control(p_values, method='bh') < 0.05

# Evaluate performance
fdr = false_discovery_rate(y_test, decisions)   # Proportion of false positives among discoveries
power = statistical_power(y_test, decisions)     # Proportion of true anomalies detected

print(f"Discoveries: {decisions.sum()}")
print(f"False Discovery Rate: {fdr:.3f}")  # Target: ≤ 0.05
print(f"Statistical Power: {power:.3f}")   # Higher is better
```

**Key metrics explained:**

- **False Discovery Rate (FDR)**: Among the observations you flagged as anomalies, what fraction are actually normal? With FDR control at 5%, this should be ≤ 0.05.
- **Statistical Power**: Among all true anomalies, what fraction did you detect? Higher power means fewer missed anomalies.

---

## Choosing a Calibration Strategy

nonconform offers several strategies for splitting data between training and calibration. The choice affects both accuracy and computational cost.

```python
from scipy.stats import false_discovery_control

from nonconform import (
    Aggregation,
    ConformalDetector,
    CrossValidation,
    JackknifeBootstrap,
    Split,
    false_discovery_rate,
    statistical_power,
)
from oddball import Dataset, load
from pyod.models.iforest import IForest

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)
base_detector = IForest(random_state=42)

# Strategy 1: Split (fastest, good for large datasets)
# Reserves 30% of training data for calibration
split_detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.3),
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Strategy 2: Cross-Validation (more data-efficient)
# Uses k-fold CV to get calibration scores from all training data
cv_detector = ConformalDetector(
    detector=base_detector,
    strategy=CrossValidation(k=5),
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Strategy 3: Jackknife+-after-Bootstrap (most robust)
# Bootstrap resampling for uncertainty quantification
jab_detector = ConformalDetector(
    detector=base_detector,
    strategy=JackknifeBootstrap(n_bootstraps=50),
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Compare strategies
for name, det in [("Split", split_detector), ("CV", cv_detector), ("JaB+", jab_detector)]:
    det.fit(x_train)
    p_values = det.predict(x_test)
    decisions = false_discovery_control(p_values, method='bh') < 0.05
    fdr = false_discovery_rate(y_test, decisions)
    power = statistical_power(y_test, decisions)
    print(f"{name:6s}: FDR={fdr:.3f}, Power={power:.3f}, Discoveries={decisions.sum()}")
```

**Which strategy should you use?**

| Scenario | Recommended Strategy |
|----------|---------------------|
| Large dataset (>5,000 samples) | `Split` — fast and sufficient |
| Medium dataset (500–5,000 samples) | `CrossValidation` — more data-efficient |
| Small dataset (<500 samples) | `JackknifeBootstrap` — most robust |
| Need maximum accuracy | `CrossValidation` or `JackknifeBootstrap` |
| Need fastest inference | `Split` |

---

## Handling Distribution Shift

If your test data comes from a different distribution than your training data (called **covariate shift**), use weighted conformal prediction:

```python
from nonconform import (
    Aggregation,
    ConformalDetector,
    Pruning,
    Split,
    false_discovery_rate,
    logistic_weight_estimator,
    statistical_power,
    weighted_false_discovery_control,
)
from oddball import Dataset, load
from pyod.models.iforest import IForest

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

# Add a weight estimator to handle distribution shift
detector = ConformalDetector(
    detector=IForest(random_state=42),
    strategy=Split(n_calib=0.3),
    aggregation=Aggregation.MEDIAN,
    weight_estimator=logistic_weight_estimator(),  # Estimates distribution shift
    seed=42
)
detector.fit(x_train)

# Get weighted p-values
p_values = detector.predict(x_test)

# Use weighted FDR control (designed for weighted conformal)
decisions = weighted_false_discovery_control(
    result=detector.last_result,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
    seed=42,
)

print(f"Discoveries: {decisions.sum()}")
print(f"FDR: {false_discovery_rate(y_test, decisions):.3f}")
print(f"Power: {statistical_power(y_test, decisions):.3f}")
```

!!! warning "When to Use Weighted Conformal"
    Use weighted conformal when:

    - Your test data is from a different time period than training data
    - There's domain shift (e.g., training on one sensor, testing on another)
    - The feature distributions differ between training and test

    The method assumes the relationship between features and anomaly status (P(Y|X)) stays the same—only the feature distribution (P(X)) changes.

---

## Using Different Detectors

nonconform works with any detector from PyOD, scikit-learn, or your own custom implementation:

```python
from scipy.stats import false_discovery_control

from nonconform import (
    Aggregation,
    ConformalDetector,
    Split,
    false_discovery_rate,
    statistical_power,
)
from oddball import Dataset, load
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=42)

# Try different base detectors
detectors = {
    'KNN': KNN(),
    'LOF': LOF(),
    'OCSVM': OCSVM()
}

for name, base_det in detectors.items():
    detector = ConformalDetector(
        detector=base_det,
        strategy=Split(n_calib=0.3),
        aggregation=Aggregation.MEDIAN,
        seed=42
    )
    detector.fit(x_train)
    p_values = detector.predict(x_test)
    decisions = false_discovery_control(p_values, method='bh') < 0.05
    fdr = false_discovery_rate(y_test, decisions)
    power = statistical_power(y_test, decisions)
    print(f"{name:6s}: FDR={fdr:.3f}, Power={power:.3f}")
```

See [Detector Compatibility](user_guide/detector_compatibility.md) for custom detector implementations.

---

## Next Steps

You now know the basics of conformal anomaly detection. To go deeper:

- **[Understanding Conformal Inference](user_guide/conformal_inference.md)** — Learn the theory behind p-values and statistical guarantees
- **[Choosing Strategies](user_guide/choosing_strategies.md)** — Detailed guidance on strategy selection
- **[FDR Control](user_guide/fdr_control.md)** — More on multiple testing and FDR control methods
- **[Examples](examples/index.md)** — Complete worked examples for different scenarios
- **[API Reference](api/index.md)** — Full documentation of all classes and functions
