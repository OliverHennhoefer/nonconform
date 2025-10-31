# Batch Evaluation

Generate evaluation batches with precise anomaly contamination control for systematic anomaly detection testing.

## Overview

The `BatchGenerator` creates evaluation batches with configurable anomaly proportion control. It supports two modes:
- **Proportional mode**: Fixed anomalies per batch (e.g., exactly 10% in each batch)
- **Probabilistic mode**: Exact global proportion across all batches

## Basic Usage

```python
from nonconform.utils.data import load, Dataset
from nonconform.utils.data.generator import BatchGenerator

# Create batch generator with proportional mode (default)
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=100,
    anomaly_proportion=0.1,  # 10% anomalies per batch
    seed=42
)

# Get training data for detector fitting
x_train = batch_gen.get_training_data()
print(f"Training data shape: {x_train.shape}")

# Generate batches (infinite for proportional mode)
for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
    anomaly_count = y_batch.sum()
    print(f"Batch {i + 1}: {x_batch.shape}, Anomalies: {anomaly_count} ({anomaly_count / len(x_batch) * 100:.1f}%)")
    if i >= 4:  # Stop after 5 batches
        break
```

## Anomaly Proportion Control Modes

### Proportional Mode (Fixed per Batch)

Ensures exact number of anomalies in each batch:

```python
# Infinite generation - exactly 15 anomalies per batch of 100
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=100,
    anomaly_proportion=0.15,
    anomaly_mode="proportional",  # Default mode
    seed=42
)

# Each batch will have exactly 15 anomalies (user controls stopping)
for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
    print(f"Anomalies: {y_batch.sum()}/100")  # Always 15
    if i >= 2:  # Stop after 3 batches
        break

# Limited generation - exactly 5 batches with 15 anomalies each
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=100,
    anomaly_proportion=0.15,
    anomaly_mode="proportional",
    n_batches=5,  # Optional limit for proportional mode
    seed=42
)

# Automatically stops after 5 batches
for x_batch, y_batch in batch_gen.generate():
    print(f"Anomalies: {y_batch.sum()}/100")  # Always 15, exactly 5 batches total
```

### Probabilistic Mode (Global Target)

Ensures exact global proportion across all batches:

```python
# Exactly 5% anomalies globally across 10 batches
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=50,
    anomaly_proportion=0.05,
    anomaly_mode="probabilistic",
    n_batches=10,  # Required for probabilistic mode
    seed=42
)

total_instances = 0
total_anomalies = 0

for x_batch, y_batch in batch_gen.generate():  # Automatically stops after n_batches
    batch_anomalies = y_batch.sum()
    total_instances += len(x_batch)
    total_anomalies += batch_anomalies
    print(f"Batch anomalies: {batch_anomalies}")

print(f"Global proportion: {total_anomalies/total_instances:.3f}")  # Exactly 0.050
```

## Integration with Conformal Detection

```python
from pyod.models.lof import LOF
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.stat import false_discovery_rate, statistical_power

# Create batch generator
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=200,
    anomaly_proportion=0.08,
    train_size=0.7,  # Use 70% of normal data for training
    seed=42
)

# Get training data and train detector
x_train = batch_gen.get_training_data()
detector = ConformalDetector(
    detector=LOF(n_neighbors=20),
    strategy=Split(n_calib=0.3)
)
detector.fit(x_train)

# Evaluate on generated batches
batch_results = []
for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
    # Get p-values
    p_values = detector.predict(x_batch)

    # Apply significance threshold
    decisions = p_values < 0.05

    # Calculate metrics
    fdr = false_discovery_rate(y_batch, decisions)
    power = statistical_power(y_batch, decisions)

    batch_results.append({
        'batch': i + 1,
        'fdr': fdr,
        'power': power,
        'detections': decisions.sum()
    })
    print(f"Batch {i + 1}: FDR={fdr:.3f}, Power={power:.3f}")

# Summary statistics
import numpy as np

mean_fdr = np.mean([r['fdr'] for r in batch_results])
mean_power = np.mean([r['power'] for r in batch_results])
print(f"Average FDR: {mean_fdr:.3f}, Average Power: {mean_power:.3f}")
```

## Advanced Configuration

### Different Datasets

```python
from nonconform.utils.data import load, Dataset

# Test with different datasets - limited generation example
datasets = [
    (lambda **kwargs: load(Dataset.SHUTTLE, **kwargs), "Shuttle"),
    (lambda **kwargs: load(Dataset.BREAST, **kwargs), "Breast Cancer"),
    (lambda **kwargs: load(Dataset.FRAUD, **kwargs), "Credit Fraud")
]

for load_func, name in datasets:
    print(f"\n{name} Dataset:")

    batch_gen = BatchGenerator(
        load_data_func=load_func,
        batch_size=100,
        anomaly_proportion=0.1,
        n_batches=3,  # Generate exactly 3 batches per dataset
        seed=42
    )

    # Check data availability
    x_train = batch_gen.get_training_data()
    print(f"  Training data: {x_train.shape}")
    print(f"  Available - Normal: {batch_gen.n_normal}, Anomaly: {batch_gen.n_anomaly}")

    # Generate all batches (automatically stops after 3)
    for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
        print(f"  Batch {i+1}: {x_batch.shape}, Anomalies: {y_batch.sum()}")
```

### Reproducibility Control

```python
# Create generator with specific seed and limited batches
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=100,
    anomaly_proportion=0.1,
    n_batches=3,  # Exactly 3 batches
    seed=42
)

# Generate initial sequence (automatically stops after 3 batches)
batch1_data = []
for x_batch, y_batch in batch_gen.generate():
    batch1_data.append((x_batch.copy(), y_batch.copy()))

# Reset generator
batch_gen.reset()

# Generate identical sequence (automatically stops after 3 batches)
batch2_data = []
for x_batch, y_batch in batch_gen.generate():
    batch2_data.append((x_batch.copy(), y_batch.copy()))

# Verify reproducibility
for i, ((x1, y1), (x2, y2)) in enumerate(zip(batch1_data, batch2_data)):
    anomalies_match = y1.sum() == y2.sum()
    print(f"Batch {i+1}: Anomaly counts match = {anomalies_match}")
```

### Performance Evaluation

```python
# Systematic evaluation across contamination levels
contamination_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
performance_results = {}

for contamination in contamination_levels:
    batch_gen = BatchGenerator(
        load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
        batch_size=150,
        anomaly_proportion=contamination,
        seed=42
    )

    # Train detector
    x_train = batch_gen.get_training_data()
    detector = ConformalDetector(
        detector=LOF(n_neighbors=20),
        strategy=Split(calib_size=0.3)
    )
    detector.fit(x_train)

    # Evaluate across multiple batches
    fdrs = []
    powers = []

    for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
        if i >= 4:  # Stop after 5 batches
            break
        p_values = detector.predict(x_batch)
        decisions = p_values < 0.05

        fdr = false_discovery_rate(y_batch, decisions)
        power = statistical_power(y_batch, decisions)

        fdrs.append(fdr)
        powers.append(power)

    performance_results[contamination] = {
        'mean_fdr': np.mean(fdrs),
        'mean_power': np.mean(powers),
        'std_fdr': np.std(fdrs),
        'std_power': np.std(powers)
    }

# Display results
print("Contamination\tFDR\t\tPower")
for contamination, results in performance_results.items():
    print(f"{contamination:.2f}\t\t{results['mean_fdr']:.3f}±{results['std_fdr']:.3f}\t{results['mean_power']:.3f}±{results['std_power']:.3f}")
```

## Generator Properties and Validation

```python
# Check generator configuration
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=100,
    anomaly_proportion=0.1,
    seed=42
)

print(f"Batch size: {batch_gen.batch_size}")
print(f"Anomaly proportion: {batch_gen.anomaly_proportion}")
print(f"Anomaly mode: {batch_gen.anomaly_mode}")
print(f"Normal instances available: {batch_gen.n_normal}")
print(f"Anomaly instances available: {batch_gen.n_anomaly}")

if batch_gen.anomaly_mode == "proportional":
    print(f"Normal per batch: {batch_gen.n_normal_per_batch}")
    print(f"Anomalies per batch: {batch_gen.n_anomaly_per_batch}")
```

## Error Handling and Validation

```python
# The generator validates parameters automatically
try:
    # Invalid batch size
    BatchGenerator(
        load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
        batch_size=0,  # Invalid
        anomaly_proportion=0.1
    )
except ValueError as e:
    print(f"Batch size error: {e}")

try:
    # Invalid anomaly proportion
    BatchGenerator(
        load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
        batch_size=100,
        anomaly_proportion=1.5  # > 1.0
    )
except ValueError as e:
    print(f"Proportion error: {e}")

try:
    # Probabilistic mode without max_batches
    BatchGenerator(
        load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
        batch_size=100,
        anomaly_proportion=0.1,
        anomaly_mode="probabilistic"
        # Missing max_batches parameter
    )
except ValueError as e:
    print(f"Mode error: {e}")
```

## Best Practices

1. **Choose appropriate batch size**: Balance between statistical power and computational efficiency
2. **Use proportional mode** for consistent per-batch evaluation
3. **Use probabilistic mode** when you need exact global contamination across all batches
4. **Set random seeds** for reproducible experiments
5. **Validate data availability** before generating many batches
6. **Reset generators** when reusing for different experiments

## Integration with FDR Control

```python
from scipy.stats import false_discovery_control

# Generate batches and apply FDR control
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=200,
    anomaly_proportion=0.1,
    seed=42
)

x_train = batch_gen.get_training_data()
detector = ConformalDetector(
    detector=LOF(n_neighbors=20),
    strategy=Split(calib_size=0.3)
)
detector.fit(x_train)

for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
    # Get p-values
    p_values = detector.predict(x_batch)

    # Apply Benjamini-Hochberg FDR control
    fdr_adjusted = false_discovery_control(p_values, method='bh')
    decisions = fdr_adjusted < 0.05

    # Calculate controlled FDR
    fdr = false_discovery_rate(y_batch, decisions)
    power = statistical_power(y_batch, decisions)

    print(f"Batch {i+1}: Controlled FDR={fdr:.3f}, Power={power:.3f}")

    if i >= 4:  # Stop after 5 batches
        break
```

This batch evaluation approach provides systematic, reproducible testing for conformal anomaly detection with precise contamination control and statistical guarantees.
