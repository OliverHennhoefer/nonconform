# Streaming Evaluation

Generate single instances for online anomaly detection evaluation with exact global anomaly proportion control.

## Overview

The `OnlineGenerator` creates deterministic streams of single instances with precise anomaly contamination. It ensures exact global proportion over a specified number of instances, making it ideal for online and streaming evaluation scenarios.

Key features:
- **Exact global proportion**: Guarantees precise anomaly ratio over total instances
- **Single instance generation**: Yields one instance at a time for streaming evaluation
- **Deterministic control**: Reproducible sequences with random seed control
- **Automatic tracking**: Manages global proportion to ensure mathematical exactness

## Basic Usage

```python
from nonconform.utils.data import load, Dataset
from nonconform.utils.data.generator import OnlineGenerator

# Create online generator with exact 2% anomalies over 1000 instances
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.02,
    n_instances=1000,
    seed=42
)

# Get training data for detector fitting
x_train = online_gen.get_training_data()
print(f"Training data shape: {x_train.shape}")

# Generate stream - exactly 20 anomalies in 1000 instances guaranteed
anomaly_count = 0
for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=1000)):
    if i < 5:  # Show first few instances
        print(f"Instance {i + 1}: {x_instance.shape}, Label: {y_label}")
    anomaly_count += y_label

print(f"Total anomalies: {anomaly_count}/1000 = {anomaly_count / 1000:.3f}")  # Exactly 0.020
```

## Exact Proportion Control

The online generator uses probabilistic tracking to ensure exact global proportions:

```python
# Test different proportions
proportions = [0.01, 0.05, 0.1, 0.15]

for prop in proportions:
    online_gen = OnlineGenerator(
        load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
        anomaly_proportion=prop,
        n_instances=500,
        seed=42
    )

    total_anomalies = 0
    for x_instance, y_label in online_gen.generate(n_instances=500):
        total_anomalies += y_label

    expected = int(500 * prop)
    actual_prop = total_anomalies / 500
    print(f"Target: {prop:.2f}, Expected: {expected}, Actual: {total_anomalies}, Proportion: {actual_prop:.3f}")
```

## Integration with Conformal Detection

```python
from pyod.models.iforest import IForest
from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.stat import false_discovery_rate, statistical_power

# Create online generator
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.03,
    n_instances=2000,
    train_size=0.6,  # Use 60% of normal data for training
    seed=42
)

# Train detector
x_train = online_gen.get_training_data()
detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(n_calib=0.3)
)
detector.fit(x_train)

# Streaming evaluation
predictions = []
labels = []
running_metrics = []

for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=2000)):
    # Get p-value for instance
    p_value = detector.predict(x_instance.reshape(1, -1))[0]

    predictions.append(p_value)
    labels.append(y_label)

    # Calculate running metrics every 100 instances
    if (i + 1) % 100 == 0:
        current_decisions = [p < 0.05 for p in predictions]
        fdr = false_discovery_rate(labels, current_decisions)
        power = statistical_power(labels, current_decisions)

        running_metrics.append({
            'instances': i + 1,
            'fdr': fdr,
            'power': power,
            'anomalies_seen': sum(labels),
            'detections': sum(current_decisions)
        })

        print(f"Instances {i + 1}: FDR={fdr:.3f}, Power={power:.3f}, Anomalies={sum(labels)}")

# Final evaluation
import numpy as np

final_decisions = [p < 0.05 for p in predictions]
final_fdr = false_discovery_rate(labels, final_decisions)
final_power = statistical_power(labels, final_decisions)
print(f"\nFinal Results: FDR={final_fdr:.3f}, Power={final_power:.3f}")
print(f"Total anomalies: {sum(labels)}/2000 = {sum(labels) / 2000:.3f}")
```

## Windowed Streaming Analysis

Analyze performance over sliding windows:

```python
# Streaming evaluation with sliding window analysis
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.05,
    n_instances=1000,
    seed=42
)

x_train = online_gen.get_training_data()
detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(calib_size=0.3)
)
detector.fit(x_train)

# Sliding window configuration
window_size = 100
window_predictions = []
window_labels = []
window_results = []

for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=1000)):
    # Get prediction
    p_value = detector.predict(x_instance.reshape(1, -1))[0]

    # Add to current window
    window_predictions.append(p_value)
    window_labels.append(y_label)

    # Process completed window
    if len(window_predictions) == window_size:
        window_decisions = [p < 0.05 for p in window_predictions]
        window_fdr = false_discovery_rate(window_labels, window_decisions)
        window_power = statistical_power(window_labels, window_decisions)

        window_results.append({
            'window_start': i - window_size + 1,
            'window_end': i,
            'fdr': window_fdr,
            'power': window_power,
            'anomalies': sum(window_labels),
            'detections': sum(window_decisions)
        })

        # Slide window (remove first half, keep second half)
        mid_point = window_size // 2
        window_predictions = window_predictions[mid_point:]
        window_labels = window_labels[mid_point:]

# Analyze window results
print("Window Analysis:")
print("Start\tEnd\tFDR\tPower\tAnomalies\tDetections")
for result in window_results[:5]:  # Show first 5 windows
    print(f"{result['window_start']}\t{result['window_end']}\t{result['fdr']:.3f}\t{result['power']:.3f}\t{result['anomalies']}\t{result['detections']}")

# Summary statistics
fdrs = [r['fdr'] for r in window_results]
powers = [r['power'] for r in window_results]
print(f"\nWindow Statistics:")
print(f"FDR: Mean={np.mean(fdrs):.3f}, Std={np.std(fdrs):.3f}")
print(f"Power: Mean={np.mean(powers):.3f}, Std={np.std(powers):.3f}")
```

## Performance and Latency Analysis

Measure per-instance processing performance:

```python
import time
import numpy as np

# Performance measurement setup
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.02,
    n_instances=1000,
    seed=42
)

x_train = online_gen.get_training_data()
detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(calib_size=0.3)
)
detector.fit(x_train)

# Measure streaming performance
latencies = []
start_time = time.time()

for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=1000)):
    instance_start = time.time()

    # Process instance (this is what would be timed in real application)
    p_value = detector.predict(x_instance.reshape(1, -1))[0]
    decision = p_value < 0.05

    instance_end = time.time()
    latencies.append(instance_end - instance_start)

    if i % 200 == 0:
        current_latency = np.mean(latencies[-200:]) if len(latencies) >= 200 else np.mean(latencies)
        print(f"Instance {i}: Avg latency = {current_latency*1000:.2f}ms")

total_time = time.time() - start_time

# Performance statistics
print(f"\nPerformance Summary:")
print(f"Total instances: 1000")
print(f"Total time: {total_time:.2f}s")
print(f"Throughput: {1000/total_time:.1f} instances/second")
print(f"Average latency: {np.mean(latencies)*1000:.2f}ms")
print(f"95th percentile latency: {np.percentile(latencies, 95)*1000:.2f}ms")
print(f"99th percentile latency: {np.percentile(latencies, 99)*1000:.2f}ms")
```

## Concept Drift Detection

Monitor for changes in performance over time:

```python
# Monitor for concept drift using performance metrics
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.04,
    n_instances=1500,
    seed=42
)

x_train = online_gen.get_training_data()
detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(calib_size=0.3)
)
detector.fit(x_train)

# Track metrics in blocks
block_size = 150
block_results = []
current_block_preds = []
current_block_labels = []

for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=1500)):
    p_value = detector.predict(x_instance.reshape(1, -1))[0]

    current_block_preds.append(p_value)
    current_block_labels.append(y_label)

    # Process completed block
    if len(current_block_preds) == block_size:
        block_decisions = [p < 0.05 for p in current_block_preds]
        block_fdr = false_discovery_rate(current_block_labels, block_decisions)
        block_power = statistical_power(current_block_labels, block_decisions)

        block_results.append({
            'block': len(block_results) + 1,
            'instances': f"{i-block_size+1}-{i}",
            'fdr': block_fdr,
            'power': block_power,
            'avg_p_value': np.mean(current_block_preds),
            'anomaly_rate': np.mean(current_block_labels)
        })

        # Reset for next block
        current_block_preds = []
        current_block_labels = []

# Analyze for drift
print("Concept Drift Analysis:")
print("Block\tInstances\tFDR\tPower\tAvg P-value\tAnomaly Rate")
for result in block_results:
    print(f"{result['block']}\t{result['instances']}\t{result['fdr']:.3f}\t{result['power']:.3f}\t{result['avg_p_value']:.3f}\t{result['anomaly_rate']:.3f}")

# Detect significant changes
fdrs = [r['fdr'] for r in block_results]
powers = [r['power'] for r in block_results]
p_values = [r['avg_p_value'] for r in block_results]

print(f"\nDrift Detection:")
print(f"FDR variation (std): {np.std(fdrs):.3f}")
print(f"Power variation (std): {np.std(powers):.3f}")
print(f"P-value variation (std): {np.std(p_values):.3f}")

# Simple drift detection based on FDR changes
if len(fdrs) > 2:
    fdr_changes = [abs(fdrs[i] - fdrs[i-1]) for i in range(1, len(fdrs))]
    if max(fdr_changes) > 0.1:
        print(f"WARNING: Potential concept drift detected (max FDR change: {max(fdr_changes):.3f})")
    else:
        print("No significant concept drift detected")
```

## Comparison with Batch Evaluation

Compare streaming vs batch evaluation approaches:

```python
from nonconform.utils.data.generator import BatchGenerator

# Streaming evaluation
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.06,
    n_instances=600,
    seed=42
)

x_train = online_gen.get_training_data()
detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(calib_size=0.3)
)
detector.fit(x_train)

# Online evaluation
online_predictions = []
online_labels = []

for x_instance, y_label in online_gen.generate(n_instances=600):
    p_value = detector.predict(x_instance.reshape(1, -1))[0]
    online_predictions.append(p_value)
    online_labels.append(y_label)

online_decisions = [p < 0.05 for p in online_predictions]
online_fdr = false_discovery_rate(online_labels, online_decisions)
online_power = statistical_power(online_labels, online_decisions)

# Batch evaluation using same data source
batch_gen = BatchGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    batch_size=100,
    anomaly_proportion=0.06,
    anomaly_mode="probabilistic",
    n_batches=6,  # 6 batches × 100 = 600 instances
    seed=42
)

batch_fdrs = []
batch_powers = []

for x_batch, y_batch in batch_gen.generate():
    p_values = detector.predict(x_batch)
    batch_decisions = p_values < 0.05

    batch_fdr = false_discovery_rate(y_batch, batch_decisions)
    batch_power = statistical_power(y_batch, batch_decisions)

    batch_fdrs.append(batch_fdr)
    batch_powers.append(batch_power)

batch_mean_fdr = np.mean(batch_fdrs)
batch_mean_power = np.mean(batch_powers)

print("Evaluation Method Comparison:")
print(f"Online Streaming:")
print(f"  FDR: {online_fdr:.3f}")
print(f"  Power: {online_power:.3f}")
print(f"  Total Anomalies: {sum(online_labels)}")

print(f"Batch Processing:")
print(f"  FDR: {batch_mean_fdr:.3f} ± {np.std(batch_fdrs):.3f}")
print(f"  Power: {batch_mean_power:.3f} ± {np.std(batch_powers):.3f}")
print(f"  Total Anomalies: {sum(sum(y_batch.values) for _, y_batch in batch_gen.generate())}")
```

## Advanced Configuration

### Different Datasets and Training Splits

```python
from nonconform.utils.data import load, Dataset

# Test with different datasets and training split ratios
configs = [
    (lambda **kwargs: load(Dataset.SHUTTLE, **kwargs), 0.5, "Shuttle"),
    (lambda **kwargs: load(Dataset.BREAST, **kwargs), 0.6, "Breast Cancer"),
    (lambda **kwargs: load(Dataset.FRAUD, **kwargs), 0.7, "Credit Fraud")
]

for load_func, train_split, name in configs:
    print(f"\n{name} Dataset (train_size={train_split}):")

    online_gen = OnlineGenerator(
        load_data_func=load_func,
        anomaly_proportion=0.03,
        n_instances=300,
        train_size=train_split,
        seed=42
    )

    x_train = online_gen.get_training_data()
    print(f"  Training data: {x_train.shape}")
    print(f"  Available for generation - Normal: {online_gen.n_normal}, Anomaly: {online_gen.n_anomaly}")

    # Quick evaluation
    total_anomalies = 0
    for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=300)):
        total_anomalies += y_label
        if i == 299:  # Last instance
            print(f"  Total anomalies: {total_anomalies}/300 = {total_anomalies / 300:.3f}")
```

### Reproducibility and Reset Functionality

```python
# Test reproducibility
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.05,
    n_instances=200,
    seed=42
)

# First run
first_run_labels = []
for x_instance, y_label in online_gen.generate(n_instances=200):
    first_run_labels.append(y_label)

first_anomalies = sum(first_run_labels)

# Reset and run again
online_gen.reset()
second_run_labels = []
for x_instance, y_label in online_gen.generate(n_instances=200):
    second_run_labels.append(y_label)

second_anomalies = sum(second_run_labels)

print("Reproducibility Test:")
print(f"First run anomalies: {first_anomalies}")
print(f"Second run anomalies: {second_anomalies}")
print(f"Results match: {first_anomalies == second_anomalies}")
print(f"Sequence identical: {first_run_labels == second_run_labels}")
```

## FDR Control in Streaming

Apply FDR control to streaming p-values:

```python
from scipy.stats import false_discovery_control

# Streaming with batch FDR control
online_gen = OnlineGenerator(
    load_data_func=lambda **kwargs: load(Dataset.SHUTTLE, **kwargs),
    anomaly_proportion=0.04,
    n_instances=500,
    seed=42
)

x_train = online_gen.get_training_data()
detector = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Split(calib_size=0.3)
)
detector.fit(x_train)

# Collect p-values for batch FDR control
batch_size = 100
all_p_values = []
all_labels = []
current_batch_p = []
current_batch_labels = []

for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=500)):
    p_value = detector.predict(x_instance.reshape(1, -1))[0]

    current_batch_p.append(p_value)
    current_batch_labels.append(y_label)

    # Process batch when full
    if len(current_batch_p) == batch_size:
        # Apply FDR control to batch
        fdr_adjusted = false_discovery_control(current_batch_p, method='bh')
        batch_decisions = fdr_adjusted < 0.05

        # Calculate controlled metrics
        batch_fdr = false_discovery_rate(current_batch_labels, batch_decisions)
        batch_power = statistical_power(current_batch_labels, batch_decisions)

        print(f"Batch {len(all_p_values)//batch_size + 1}: FDR={batch_fdr:.3f}, Power={batch_power:.3f}, Detections={sum(batch_decisions)}")

        all_p_values.extend(current_batch_p)
        all_labels.extend(current_batch_labels)
        current_batch_p = []
        current_batch_labels = []

# Overall FDR control
overall_fdr_adjusted = false_discovery_control(all_p_values, method='bh')
overall_decisions = overall_fdr_adjusted < 0.05
overall_fdr = false_discovery_rate(all_labels, overall_decisions)
overall_power = statistical_power(all_labels, overall_decisions)

print(f"\nOverall Results with FDR Control:")
print(f"FDR: {overall_fdr:.3f}")
print(f"Power: {overall_power:.3f}")
print(f"Total detections: {sum(overall_decisions)}")
print(f"Total anomalies: {sum(all_labels)}")
```

## Best Practices

1. **Use appropriate n_instances**: Set based on your evaluation requirements and computational constraints
2. **Monitor global proportion**: The generator guarantees exact proportions mathematically
3. **Apply proper FDR control**: Use batch-wise FDR control for streaming scenarios
4. **Track performance metrics**: Monitor latency and throughput for operational insights
5. **Reset for reproducibility**: Use reset() when repeating experiments
6. **Consider concept drift**: Monitor performance changes over time windows

## Memory and Computational Efficiency

The online generator is designed for efficiency:
- **Low memory footprint**: Only stores necessary data for proportion tracking
- **Fast instance generation**: Minimal overhead per instance
- **Deterministic behavior**: Reproducible results with proper seed management
- **Automatic validation**: Built-in parameter and proportion checking

This streaming evaluation approach enables rigorous online testing of conformal anomaly detection with exact statistical control and efficient resource utilization.
