import logging
from collections.abc import Callable
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import set_params
from pyod.models.base import BaseDetector


class JackknifeBootstrap(BaseStrategy):
    """Implements Jackknife+-after-Bootstrap (JaB+) conformal anomaly detection.

    This strategy implements the JaB+ method which provides predictive inference
    for ensemble models trained on bootstrap samples. The key insight is that
    JaB+ uses the out-of-bag (OOB) samples from bootstrap iterations to compute
    calibration scores without requiring additional model training.

    The method works as follows:
    1. Generate B bootstrap samples from the training data
    2. Train B models, one on each bootstrap sample
    3. For each original training sample, use the models where that sample was
       out-of-bag to compute calibration scores
    4. Train a final aggregated model on all data for prediction
    5. Use the calibration scores to convert predictions to p-values

    This provides the coverage guarantees of Jackknife+ but with the computational
    efficiency of bootstrap methods.

    Note: JaB+ is only valid with plus=False (single final model), not with
    ensemble prediction (plus=True).

    Attributes
    ----------
        _n_bootstraps (int): Number of bootstrap iterations
        _aggregation_method (Aggregation): How to aggregate OOB predictions
        _detector_list (list[BaseDetector]): List containing the final trained detector
        _calibration_set (list[float]): List of calibration scores from JaB+ procedure
        _calibration_ids (list[int]): Indices of samples used for calibration
        _bootstrap_models (list[BaseDetector]): Models trained on each bootstrap sample
        _oob_indices (list[set[int]]): Out-of-bag indices for each bootstrap iteration
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        aggregation_method: Aggregation = Aggregation.MEAN,
    ):
        """Initialize the Bootstrap (JaB+) strategy.

        Args:
            n_bootstraps (int, optional): Number of bootstrap iterations.
                Defaults to 100.
            aggregation_method (Aggregation, optional): Method to aggregate out-of-bag
                predictions. Options are Aggregation.MEAN or Aggregation.MEDIAN.
                Defaults to Aggregation.MEAN.

        Raises
        ------
            ValueError: If aggregation_method is not a valid Aggregation enum value.
            ValueError: If n_bootstraps is less than 1.
        """
        # JaB+ is only valid for plus=False
        super().__init__(plus=False)

        if n_bootstraps < 1:
            raise ValueError("Number of bootstraps must be at least 1.")
        if not isinstance(aggregation_method, Aggregation):
            raise ValueError("aggregation_method must be an Aggregation enum value")
        if aggregation_method not in [Aggregation.MEAN, Aggregation.MEDIAN]:
            raise ValueError(
                "aggregation_method must be Aggregation.MEAN or Aggregation.MEDIAN"
            )

        self._n_bootstraps: int = n_bootstraps
        self._aggregation_method: Aggregation = aggregation_method

        self._detector_list: list[BaseDetector] = []
        self._calibration_set: list[float] = []
        self._calibration_ids: list[int] = []

        # Internal state for JaB+ computation
        self._bootstrap_models: list[BaseDetector] = []
        self._oob_indices: list[set[int]] = []

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int | None = None,
        weighted: bool = False,
        iteration_callback: Callable[[int, np.ndarray], None] | None = None,
    ) -> tuple[list[BaseDetector], list[float]]:
        """Fit and calibrate using Jackknife+-after-Bootstrap method.

        This method implements the JaB+ algorithm:
        1. Generate bootstrap samples and train models
        2. For each sample, compute out-of-bag predictions
        3. Aggregate OOB predictions to get calibration scores
        4. Train final model on all data

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Input data matrix of shape
                (n_samples, n_features).
            detector (BaseDetector): The base anomaly detector to be used.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            weighted (bool, optional): Not used in JaB+ method. Defaults to False.
            iteration_callback (Callable[[int, np.ndarray], None], optional):
                Optional callback function that gets called after each bootstrap
                iteration with the iteration number and current calibration scores.
                Defaults to None.

        Returns
        -------
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * List with single trained detector model
                * List of calibration scores from JaB+ procedure
        """
        n_samples = len(x)
        logger = get_logger("strategy.bootstrap")
        generator = np.random.default_rng(seed)

        logger.info(
            f"Bootstrap (JaB+) Configuration:\n"
            f"  • Data: {n_samples:,} total samples\n"
            f"  • Bootstrap iterations: {self._n_bootstraps:,}\n"
            f"  • Aggregation method: {self._aggregation_method}"
        )

        # Step 1: Generate bootstrap samples and train models
        self._bootstrap_models = []
        self._oob_indices = []

        for i in tqdm(
            range(self._n_bootstraps),
            desc=f"Bootstrap training ({self._n_bootstraps} iterations)",
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            # Generate bootstrap sample (sample with replacement)
            bootstrap_indices = generator.choice(
                n_samples, size=n_samples, replace=True
            )

            # Track out-of-bag samples
            in_bag_set = set(bootstrap_indices)
            oob_indices = set(range(n_samples)) - in_bag_set
            self._oob_indices.append(oob_indices)

            # Train model on bootstrap sample
            model = copy(detector)
            model = set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[bootstrap_indices])
            self._bootstrap_models.append(deepcopy(model))

        # Step 2: Compute out-of-bag calibration scores
        oob_scores = self._compute_oob_scores(x)

        # Call iteration callback if provided
        if iteration_callback is not None:
            iteration_callback(self._n_bootstraps, oob_scores)

        self._calibration_set = oob_scores.tolist()
        self._calibration_ids = list(range(n_samples))

        # Step 3: Train final model on all data
        final_model = copy(detector)
        final_model = set_params(
            final_model,
            seed=seed,
            random_iteration=True,
            iteration=self._n_bootstraps,
        )
        final_model.fit(x)
        self._detector_list = [deepcopy(final_model)]

        logger.info(
            f"JaB+ calibration completed with {len(self._calibration_set)} scores"
        )

        return self._detector_list, self._calibration_set

    def _compute_oob_scores(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute out-of-bag calibration scores for JaB+ method.

        For each sample, this method:
        1. Finds all bootstrap models where the sample was out-of-bag
        2. Uses those models to predict the sample's anomaly score
        3. Aggregates the predictions using the specified aggregation method

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Input data matrix.

        Returns
        -------
            np.ndarray: Array of calibration scores for each sample.

        Raises
        ------
            ValueError: If a sample has no out-of-bag predictions (very unlikely).
        """
        n_samples = len(x)
        oob_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Find models where sample i was out-of-bag
            oob_model_indices = []
            for j, oob_set in enumerate(self._oob_indices):
                if i in oob_set:
                    oob_model_indices.append(j)

            if not oob_model_indices:
                # This is extremely unlikely with bootstrap sampling
                raise ValueError(
                    f"Sample {i} has no out-of-bag predictions. "
                    "Consider increasing n_bootstraps."
                )

            # Get predictions from OOB models
            sample_predictions = []
            for model_idx in oob_model_indices:
                model = self._bootstrap_models[model_idx]
                pred = model.decision_function(x[i].reshape(1, -1))[0]
                sample_predictions.append(pred)

            # Aggregate predictions
            if self._aggregation_method == Aggregation.MEAN:
                oob_scores[i] = np.mean(sample_predictions)
            else:  # Aggregation.MEDIAN
                oob_scores[i] = np.median(sample_predictions)

        return oob_scores

    @property
    def calibration_ids(self) -> list[int]:
        """Returns the list of indices used for calibration.

        In JaB+, all original training samples contribute to calibration
        through the out-of-bag mechanism.

        Returns
        -------
            list[int]: List of integer indices (0 to n_samples-1).
        """
        return self._calibration_ids

    @property
    def n_bootstraps(self) -> int:
        """Returns the number of bootstrap iterations."""
        return self._n_bootstraps

    @property
    def aggregation_method(self) -> Aggregation:
        """Returns the aggregation method used for OOB predictions."""
        return self._aggregation_method
