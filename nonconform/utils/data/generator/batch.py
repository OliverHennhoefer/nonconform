from collections.abc import Callable, Iterator
from typing import Literal

import pandas as pd

from nonconform.utils.func.logger import get_logger

from .base import BaseDataGenerator

logger = get_logger("nonconform.utils.data.generator.batch")


class BatchGenerator(BaseDataGenerator):
    """Generate batches with configurable anomaly contamination.

    Parameters
    ----------
    load_data_func : Callable[[], pd.DataFrame]
        Function from nonconform.utils.data.load (e.g., load_shuttle).
    batch_size : int
        Number of instances per batch.
    anomaly_proportion : float
        Target proportion of anomalies (0.0 to 1.0).
    anomaly_mode : {"proportional", "probabilistic"}, default="proportional"
        How to control anomaly proportions.
    n_batches : int, optional
        Number of batches to generate.
        - Required for "probabilistic" mode
        - Optional for "proportional" mode (if None, generates indefinitely)
    train_size : float, default=0.5
        Proportion of normal instances to use for training.
    seed : int, optional
        Seed for random number generator.

    Examples:
    --------
    >>> from nonconform.utils.data.load import load_shuttle
    >>> from nonconform.utils.data.generator import BatchGenerator
    >>>
    >>> # Proportional mode - 10% anomalies per batch
    >>> batch_gen = BatchGenerator(
    ...     load_data_func=load_shuttle, batch_size=100, anomaly_proportion=0.1, seed=42
    ... )
    >>>
    >>> # Proportional mode with limited batches - 10% anomalies for exactly 5 batches
    >>> batch_gen = BatchGenerator(
    ...     load_data_func=load_shuttle,
    ...     batch_size=100,
    ...     anomaly_proportion=0.1,
    ...     anomaly_mode="proportional",
    ...     n_batches=5,
    ...     seed=42,
    ... )
    >>>
    >>> # Probabilistic mode - 5% anomalies across 10 batches
    >>> batch_gen = BatchGenerator(
    ...     load_data_func=load_shuttle,
    ...     batch_size=100,
    ...     anomaly_proportion=0.05,
    ...     anomaly_mode="probabilistic",
    ...     n_batches=10,
    ...     seed=42,
    ... )
    >>>
    >>> # Get training data
    >>> x_train = batch_gen.get_training_data()
    >>>
    >>> # Generate batches (infinite for proportional mode)
    >>> for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
    ...     print(f"Batch: {x_batch.shape}, Anomalies: {y_batch.sum()}")
    ...     if i >= 4:  # Stop after 5 batches
    ...         break
    >>>
    >>> # Proportional mode with n_batches - automatic stopping after 5 batches
    >>> for x_batch, y_batch in batch_gen.generate():
    ...     print(f"Batch: {x_batch.shape}, Anomalies: {y_batch.sum()}")
    >>>
    >>> # Probabilistic mode - automatic stopping after n_batches
    >>> for x_batch, y_batch in batch_gen.generate():
    ...     print(f"Batch: {x_batch.shape}, Anomalies: {y_batch.sum()}")
    """

    def __init__(
        self,
        load_data_func: Callable[[], pd.DataFrame],
        batch_size: int,
        anomaly_proportion: float,
        anomaly_mode: Literal["proportional", "probabilistic"] = "proportional",
        n_batches: int | None = None,
        train_size: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the batch generator."""
        self.batch_size = batch_size

        # Validate batch size
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Initialize base class
        super().__init__(
            load_data_func=load_data_func,
            anomaly_proportion=anomaly_proportion,
            anomaly_mode=anomaly_mode,
            n_batches=n_batches,
            train_size=train_size,
            seed=seed,
        )

        # Calculate anomaly count per batch for proportional mode
        if anomaly_mode == "proportional":
            self.n_anomaly_per_batch = int(batch_size * anomaly_proportion)
            self.n_normal_per_batch = batch_size - self.n_anomaly_per_batch

            # Warn if anomaly proportion truncates to zero
            if anomaly_proportion > 0 and self.n_anomaly_per_batch == 0:
                min_batch_size = int(1 / anomaly_proportion)
                logger.warning(
                    f"Batch size {batch_size} with proportion {anomaly_proportion:.2%} "
                    f"results in 0 anomalies per batch due to truncation. "
                    f"Consider using batch_size >= {min_batch_size} "
                    f"or use anomaly_mode='probabilistic' for exact global proportion."
                )

            self._validate_batch_config()

    def _validate_batch_config(self) -> None:
        """Validate batch-specific configuration."""
        if self.anomaly_mode == "proportional":
            if self.n_normal_per_batch > self.n_normal:
                raise ValueError(
                    f"Not enough normal instances ({self.n_normal}) for "
                    f"batch size ({self.n_normal_per_batch} needed per batch)"
                )
            if self.n_anomaly_per_batch > self.n_anomaly:
                raise ValueError(
                    f"Not enough anomaly instances ({self.n_anomaly}) for "
                    f"batch size ({self.n_anomaly_per_batch} needed per batch)"
                )

    def generate(self) -> Iterator[tuple[pd.DataFrame, pd.Series]]:
        """Generate batches with mixed normal and anomalous instances.

        - For proportional mode: generates batches indefinitely if n_batches=None,
          or exactly n_batches batches if specified in constructor
        - For probabilistic mode: generates exactly n_batches batches
          (required in constructor)

        Yields:
        ------
        x_batch : pd.DataFrame
            Feature matrix for the batch.
        y_batch : pd.Series
            Labels for the batch (0=normal, 1=anomaly).
        """
        batch_count = 0

        # Determine stopping condition based on mode and n_batches
        def _should_continue():
            if self.anomaly_mode == "proportional":
                # Proportional: stop when n_batches reached (if specified),
                # otherwise infinite
                return self.n_batches is None or batch_count < self.n_batches
            else:
                # Probabilistic: always stop at n_batches (required)
                return batch_count < self.n_batches

        while _should_continue():
            if self.anomaly_mode == "proportional":
                # Proportional mode: exact number of anomalies per batch
                batch_data = []
                batch_labels = []

                # Generate exact number of normal instances
                for _ in range(self.n_normal_per_batch):
                    instance, label = self._sample_instance(False)
                    batch_data.append(instance)
                    batch_labels.append(label)

                # Generate exact number of anomaly instances
                for _ in range(self.n_anomaly_per_batch):
                    instance, label = self._sample_instance(True)
                    batch_data.append(instance)
                    batch_labels.append(label)

                # Combine and shuffle
                x_batch = pd.concat(batch_data, axis=0, ignore_index=True)
                y_batch = pd.Series(batch_labels, dtype=int)

                # Shuffle the batch to mix normal and anomalous instances
                shuffle_idx = self.rng.permutation(self.batch_size)
                x_batch = x_batch.iloc[shuffle_idx].reset_index(drop=True)
                y_batch = y_batch.iloc[shuffle_idx].reset_index(drop=True)

            else:  # probabilistic mode
                # Probabilistic mode: use global tracking to ensure exact proportion
                batch_data = []
                batch_labels = []

                # Generate instances for this batch
                for _ in range(self.batch_size):
                    is_anomaly = self._should_generate_anomaly()
                    instance, label = self._sample_instance(is_anomaly)

                    batch_data.append(instance)
                    batch_labels.append(label)

                    # Update tracking
                    self._current_anomalies += label
                    self._items_generated += 1

                # Combine into batch
                x_batch = pd.concat(batch_data, axis=0, ignore_index=True)
                y_batch = pd.Series(batch_labels, dtype=int)

            yield x_batch, y_batch
            batch_count += 1
