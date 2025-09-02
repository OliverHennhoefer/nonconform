"""Abstract base class for data generators with anomaly contamination control."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
import pandas as pd


class BaseDataGenerator(ABC):
    """Abstract base class for data generators with anomaly contamination.

    This class defines the interface for generating data with controlled anomaly
    contamination. It supports both batch and online generation modes with
    different anomaly proportion control strategies.

    Parameters:
    load_data_func : Callable[[], pd.DataFrame]
        Function from nonconform.utils.data.load (e.g., load_shuttle, load_breast).
    anomaly_proportion : float
        Target proportion of anomalies (0.0 to 1.0).
    anomaly_mode : {"proportional", "probabilistic"}, default="proportional"
        How to control anomaly proportions:
        - "proportional": Fixed proportion per batch/instance
        - "probabilistic": Probabilistic with global target over all items
    n_batches : int, optional
        Number of batches/instances for "probabilistic" mode.
        Required when anomaly_mode="probabilistic".
    train_size : float, default=0.5
        Proportion of normal instances to use for training.
    seed : int, optional
        Seed for random number generator.

    Attributes:
    x_train : pd.DataFrame
        Training data (normal instances only).
    x_normal : pd.DataFrame
        Normal instances for generation.
    x_anomaly : pd.DataFrame
        Anomalous instances for generation.
    n_normal : int
        Number of normal instances available.
    n_anomaly : int
        Number of anomalous instances available.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(
        self,
        load_data_func: Callable[[], pd.DataFrame],
        anomaly_proportion: float,
        anomaly_mode: Literal["proportional", "probabilistic"] = "proportional",
        n_batches: int | None = None,
        train_size: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the base data generator."""
        self.load_data_func = load_data_func
        self.anomaly_proportion = anomaly_proportion
        self.anomaly_mode = anomaly_mode
        self.n_batches = n_batches
        self.train_size = train_size
        self.seed = seed

        # Initialize random number generator
        self.rng = np.random.default_rng(seed)

        # Validate configuration
        self._validate_config()

        # Load and prepare data
        self._prepare_data()

        # Initialize anomaly tracking for probabilistic mode
        if anomaly_mode == "probabilistic":
            self._init_probabilistic_tracking()

    def _validate_config(self) -> None:
        """Validate the generator configuration."""
        if not 0 <= self.anomaly_proportion <= 1:
            raise ValueError(
                f"anomaly_proportion must be between 0 and 1, "
                f"got {self.anomaly_proportion}"
            )

        if not 0 < self.train_size < 1:
            raise ValueError(
                f"train_size must be between 0 and 1, got {self.train_size}"
            )

        if self.anomaly_mode not in ["proportional", "probabilistic"]:
            raise ValueError(
                f"anomaly_mode must be 'proportional' or 'probabilistic', "
                f"got {self.anomaly_mode}"
            )

        if self.anomaly_mode == "probabilistic" and self.n_batches is None:
            raise ValueError(
                "n_batches must be specified when anomaly_mode='probabilistic'"
            )

        if self.n_batches is not None and self.n_batches <= 0:
            raise ValueError(f"n_batches must be positive, got {self.n_batches}")

    def _prepare_data(self) -> None:
        """Load and prepare data for generation."""
        # Load complete dataset
        df = self.load_data_func()

        # Separate normal and anomalous instances
        normal_mask = df["Class"] == 0
        df_normal = df[normal_mask]
        df_anomaly = df[~normal_mask]

        if len(df_normal) == 0:
            raise ValueError("No normal instances found in dataset")
        if len(df_anomaly) == 0:
            raise ValueError("No anomalous instances found in dataset")

        # Split normal data into train and test
        n_train = int(len(df_normal) * self.train_size)
        train_idx = self.rng.choice(len(df_normal), size=n_train, replace=False)
        test_normal_mask = np.ones(len(df_normal), dtype=bool)
        test_normal_mask[train_idx] = False

        # Create datasets
        self.x_train = df_normal.iloc[train_idx].drop(columns=["Class"])
        self.x_normal = df_normal.iloc[test_normal_mask].drop(columns=["Class"])
        self.x_anomaly = df_anomaly.drop(columns=["Class"])

        # Store dataset sizes
        self.n_normal = len(self.x_normal)
        self.n_anomaly = len(self.x_anomaly)

    def _init_probabilistic_tracking(self) -> None:
        """Initialize tracking for probabilistic anomaly mode."""
        # For probabilistic mode, we need to track total items across all
        # batches/instances
        # to ensure exact global proportion
        if hasattr(self, "batch_size"):
            # Batch mode: calculate total instances across all batches
            total_instances = self.n_batches * self.batch_size
        else:
            # Online mode: n_batches is the total instances
            total_instances = self.n_batches

        self._target_anomalies = int(total_instances * self.anomaly_proportion)
        self._current_anomalies = 0
        self._items_generated = 0

    def get_training_data(self) -> pd.DataFrame:
        """Get training data (normal instances only).

        Returns:
        pd.DataFrame
            Training data without anomalies.
        """
        return self.x_train

    def reset(self) -> None:
        """Reset the generator to initial state."""
        self.rng = np.random.default_rng(self.seed)
        if self.anomaly_mode == "probabilistic":
            self._current_anomalies = 0
            self._items_generated = 0

    def _should_generate_anomaly(self) -> bool:
        """Determine if next item should be anomaly based on mode."""
        # This method should not be used directly for proportional mode in batch
        # generators
        # It's primarily for probabilistic mode or online generators
        if self.anomaly_mode == "probabilistic":
            # Calculate remaining items and anomalies needed
            if hasattr(self, "batch_size"):
                # Batch mode: total instances = n_batches * batch_size
                total_instances = self.n_batches * self.batch_size
            else:
                # Online mode: n_batches is total instances
                total_instances = self.n_batches

            remaining_items = total_instances - self._items_generated
            remaining_anomalies = self._target_anomalies - self._current_anomalies

            if remaining_items <= 0:
                return False
            if remaining_anomalies <= 0:
                return False
            if remaining_anomalies >= remaining_items:
                return True

            # Probability based on remaining targets
            return self.rng.random() < (remaining_anomalies / remaining_items)

        # For proportional mode in online generators
        return self.rng.random() < self.anomaly_proportion

    def _sample_instance(self, is_anomaly: bool) -> tuple[pd.DataFrame, int]:
        """Sample a single instance.

        Parameters
        is_anomaly : bool
            Whether to sample an anomaly or normal instance.

        Returns:
        tuple[pd.DataFrame, int]
            Instance data and label (0=normal, 1=anomaly).
        """
        if is_anomaly:
            idx = self.rng.integers(0, self.n_anomaly)
            instance = self.x_anomaly.iloc[[idx]].reset_index(drop=True)
            label = 1
        else:
            idx = self.rng.integers(0, self.n_normal)
            instance = self.x_normal.iloc[[idx]].reset_index(drop=True)
            label = 0

        return instance, label

    @abstractmethod
    def generate(self, **kwargs) -> Iterator[Any]:
        """Generate data items.

        This method must be implemented by subclasses to define
        the specific generation behavior (batch vs online).
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        return (
            f"{self.__class__.__name__}("
            f"n_normal={self.n_normal}, "
            f"n_anomaly={self.n_anomaly}, "
            f"anomaly_proportion={self.anomaly_proportion}, "
            f"anomaly_mode='{self.anomaly_mode}')"
        )
