from nonconform.utils.data.generator import BatchGenerator, OnlineGenerator
from nonconform.utils.data.load import load_breast, load_shuttle


def demonstrate_batch_generation():
    """Demonstrate batch generation with both anomaly modes."""
    print("=== Batch Generation Examples ===")

    # Proportional mode - fixed 10% anomalies per batch
    print("1. Proportional Mode (10% anomalies per batch):")
    batch_gen = BatchGenerator(
        load_data_func=load_shuttle,
        batch_size=100,
        anomaly_proportion=0.1,
        anomaly_mode="proportional",
        n_batches=3,
        train_size=0.5,
        random_state=42,
    )

    print(f"   Generator: {batch_gen}")
    print(f"   Training data: {batch_gen.get_training_data().shape}")

    for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
        anomaly_count = y_batch.sum()
        print(
            f"   Batch {i+1}: {x_batch.shape}, Anomalies: {anomaly_count} ({anomaly_count/len(x_batch)*100:.1f}%)"
        )

    # Probabilistic mode - global target across all batches
    print("\n2. Probabilistic Mode (5% anomalies globally across 10 batches):")
    batch_gen_prob = BatchGenerator(
        load_data_func=load_breast,
        batch_size=50,
        anomaly_proportion=0.05,
        anomaly_mode="probabilistic",
        n_batches=10,
        random_state=42,
    )

    print(f"   Generator: {batch_gen_prob}")
    total_instances = 0
    total_anomalies = 0

    for i, (x_batch, y_batch) in enumerate(batch_gen_prob.generate()):
        anomaly_count = y_batch.sum()
        total_instances += len(x_batch)
        total_anomalies += anomaly_count
        if i < 3:  # Show first 3 batches
            print(
                f"   Batch {i+1}: {x_batch.shape}, Anomalies: {anomaly_count} ({anomaly_count/len(x_batch)*100:.1f}%)"
            )

    print("   ... (7 more batches)")
    print(
        f"   Total: {total_instances} instances, {total_anomalies} anomalies ({total_anomalies/total_instances*100:.1f}%)"
    )
    print()


def demonstrate_online_generation():
    """Demonstrate online generation with exact global anomaly proportion."""
    print("=== Online Generation Examples ===")

    # Online generator always uses probabilistic mode for exact global proportion
    print("Online Generator (exactly 2% anomalies over 1000 instances):")
    online_gen = OnlineGenerator(
        load_data_func=load_shuttle,
        anomaly_proportion=0.02,
        n_instances=1000,
        random_state=42,
    )

    print(f"   Generator: {online_gen}")
    print(f"   Training data: {online_gen.get_training_data().shape}")
    print(f"   Expected anomalies in 1000 instances: {int(1000 * 0.02)} (exactly)")

    # Generate all 1000 instances
    anomaly_count = 0
    for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=1000)):
        anomaly_count += y_label
        if i < 5:  # Show first 5 instances
            print(f"   Instance {i+1}: {x_instance.shape}, Label: {y_label}")

    print("   ... (995 more instances)")
    print(
        f"   Actual anomalies in 1000 instances: {anomaly_count} (exactly {int(1000 * 0.02)} as guaranteed)"
    )

    # Smaller example to show exact control
    print("\nSmaller Example (exactly 1% anomalies over 100 instances):")
    online_gen_small = OnlineGenerator(
        load_data_func=load_breast,
        anomaly_proportion=0.01,
        n_instances=100,
        random_state=42,
    )

    anomaly_count = 0
    for i, (x_instance, y_label) in enumerate(
        online_gen_small.generate(n_instances=100)
    ):
        anomaly_count += y_label
        if i < 5:  # Show first 5 instances
            print(f"   Instance {i+1}: {x_instance.shape}, Label: {y_label}")

    print("   ... (95 more instances)")
    print(
        f"   Total anomalies in 100 instances: {anomaly_count} (exactly {int(100 * 0.01)} as guaranteed)"
    )
    print()


def demonstrate_integration_workflow():
    """Demonstrate complete workflow with training and generation."""
    print("=== Complete Workflow Example ===")

    # Create batch generator
    batch_gen = BatchGenerator(
        load_data_func=load_shuttle,
        batch_size=200,
        anomaly_proportion=0.08,
        anomaly_mode="proportional",
        n_batches=3,
        train_size=0.7,
        random_state=42,
    )

    # Get training data for detector
    x_train = batch_gen.get_training_data()
    print(f"Training data shape: {x_train.shape}")
    print(f"Training data sample means: {x_train.mean().head()}")

    # Simulate detector training (normally you'd train a PyOD detector here)
    print("Training detector on normal data...")

    # Generate evaluation batches
    print("\nGenerating evaluation batches:")
    for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
        anomaly_count = y_batch.sum()
        batch_mean = x_batch.mean().mean()
        print(
            f"   Batch {i+1}: {x_batch.shape}, Anomalies: {anomaly_count}, Mean features: {batch_mean:.3f}"
        )

    print()


def demonstrate_different_datasets():
    """Show generators working with different datasets."""
    print("=== Different Datasets Example ===")

    datasets = [(load_shuttle, "Shuttle"), (load_breast, "Breast Cancer")]

    for load_func, name in datasets:
        print(f"{name} Dataset:")

        # Create online generator with exact proportion over 200 instances
        online_gen = OnlineGenerator(
            load_data_func=load_func,
            anomaly_proportion=0.05,
            n_instances=200,
            random_state=42,
        )

        x_train = online_gen.get_training_data()
        print(f"   Training data: {x_train.shape}")
        print(
            f"   Available for generation - Normal: {online_gen.n_normal}, Anomaly: {online_gen.n_anomaly}"
        )
        print(f"   Expected anomalies in 200 instances: {int(200 * 0.05)} (exactly)")

        # Generate all 200 instances
        sample_anomalies = 0
        for i, (x_instance, y_label) in enumerate(online_gen.generate(n_instances=200)):
            sample_anomalies += y_label

        print(
            f"   Actual anomalies in 200 instances: {sample_anomalies} (exactly {int(200 * 0.05)} as guaranteed)"
        )
        print()


def main():
    """Run all demonstrations."""
    print("Data Generator Examples")
    print("=" * 50)

    demonstrate_batch_generation()
    demonstrate_online_generation()
    demonstrate_integration_workflow()
    demonstrate_different_datasets()

    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
