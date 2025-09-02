from collections.abc import Callable, Iterator

import pandas as pd

from .base import BaseDataGenerator


class OnlineGenerator(BaseDataGenerator):
    """Generate single instances with probabilistic anomaly contamination for streaming.

    Online generators use probabilistic anomaly control to ensure exact global
    proportion over a specified number of instances.

    Parameters
    ----------
    load_data_func : Callable[[], pd.DataFrame]
        Function from nonconform.utils.data.load (e.g., load_shuttle).
    anomaly_proportion : float
        Target proportion of anomalies (0.0 to 1.0).
    n_instances : int
        Number of instances to ensure exact global proportion.
    train_size : float, default=0.5
        Proportion of normal instances to use for training.
    seed : int, optional
        Seed for random number generator.

    Examples
    --------
    >>> from nonconform.utils.data.load import load_shuttle
    >>> from nonconform.utils.data.generator import OnlineGenerator
    >>>
    >>> # Exactly 1% anomalies over 1000 instances
    >>> online_gen = OnlineGenerator(
    ...     load_data_func=load_shuttle,
    ...     anomaly_proportion=0.01,
    ...     n_instances=1000,
    ...     seed=42,
    ... )
    >>>
    >>> # Get training data
    >>> x_train = online_gen.get_training_data()
    >>>
    >>> # Generate instances - exactly 10 anomalies in 1000 instances
    >>> for x_instance, y_label in online_gen.generate(n_instances=1000):
    ...     print(f"Instance: {x_instance.shape}, Label: {y_label}")
    """

    def __init__(
        self,
        load_data_func: Callable[[], pd.DataFrame],
        anomaly_proportion: float,
        n_instances: int,
        train_size: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the online generator."""
        # Initialize base class with probabilistic mode
        super().__init__(
            load_data_func=load_data_func,
            anomaly_proportion=anomaly_proportion,
            anomaly_mode="probabilistic",
            n_batches=n_instances,
            train_size=train_size,
            seed=seed,
        )

    def generate(
        self, n_instances: int | None = None
    ) -> Iterator[tuple[pd.DataFrame, int]]:
        """Generate stream of single instances with exact anomaly proportion.

        Parameters
        ----------
        n_instances : int, optional
            Number of instances to generate. If None, generates up to max_instances.

        Yields
        ------
        x_instance : pd.DataFrame
            Single instance feature vector.
        y_label : int
            Label for the instance (0=normal, 1=anomaly).
        """
        # Default to n_instances if not specified
        if n_instances is None:
            n_instances = self.n_batches

        # Validate we don't exceed n_instances
        if n_instances > self.n_batches:
            raise ValueError(
                f"Requested {n_instances} instances exceeds n_instances "
                f"({self.n_batches}). Global proportion cannot be guaranteed."
            )

        instance_count = 0

        while instance_count < n_instances:
            # Determine if this instance should be anomaly using global tracking
            is_anomaly = self._should_generate_anomaly()

            # Sample instance
            instance, label = self._sample_instance(is_anomaly)

            # Update tracking
            self._current_anomalies += label
            self._items_generated += 1

            yield instance, label
            instance_count += 1
