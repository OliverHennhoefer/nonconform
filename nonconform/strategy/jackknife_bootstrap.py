import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial

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
        _oob_mask (np.ndarray): Boolean matrix of shape (n_bootstraps, n_samples)
            indicating out-of-bag status
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
        super().__init__(plus=False)

        if n_bootstraps < 1:
            raise ValueError("Number of bootstraps must be at least 1.")
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
        self._oob_mask: np.ndarray = np.array([])

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int | None = None,
        weighted: bool = False,
        iteration_callback: Callable[[int, np.ndarray], None] | None = None,
        n_jobs: int | None = None,
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
            n_jobs (int, optional): Number of parallel jobs for bootstrap
                training. If None, uses sequential processing. Defaults to None.

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

        # Step 1: Pre-allocate data structures and generate bootstrap samples
        self._bootstrap_models = [None] * self._n_bootstraps
        self._oob_mask = np.zeros((self._n_bootstraps, n_samples), dtype=bool)

        # Generate all bootstrap indices at once for better memory locality
        all_bootstrap_indices = generator.choice(
            n_samples, size=(self._n_bootstraps, n_samples), replace=True
        )

        # Pre-compute OOB mask efficiently
        for i in range(self._n_bootstraps):
            bootstrap_indices = all_bootstrap_indices[i]
            in_bag_mask = np.zeros(n_samples, dtype=bool)
            in_bag_mask[bootstrap_indices] = True
            self._oob_mask[i] = ~in_bag_mask

        # Train models (with optional parallelization)
        if n_jobs is None or n_jobs == 1:
            # Sequential training
            for i in tqdm(
                range(self._n_bootstraps),
                desc=f"Bootstrap training ({self._n_bootstraps} iterations)",
                disable=not logger.isEnabledFor(logging.INFO),
            ):
                bootstrap_indices = all_bootstrap_indices[i]
                model = self._train_single_model(
                    detector, x, bootstrap_indices, seed, i
                )
                self._bootstrap_models[i] = model
        else:
            # Parallel training
            self._train_models_parallel(
                detector, x, all_bootstrap_indices, seed, n_jobs, logger
            )

        # Step 2: Compute out-of-bag calibration scores
        oob_scores = self._compute_oob_scores(x)

        # Call iteration callback if provided
        if iteration_callback is not None:
            iteration_callback(self._n_bootstraps, oob_scores)

        self._calibration_set = oob_scores.tolist()
        self._calibration_ids = list(range(n_samples))

        # Step 3: Train final model on all data
        final_model = deepcopy(detector)
        final_model = set_params(
            final_model,
            seed=seed,
            random_iteration=True,
            iteration=self._n_bootstraps,
        )
        final_model.fit(x)
        self._detector_list = [final_model]

        logger.info(
            f"JaB+ calibration completed with {len(self._calibration_set)} scores"
        )

        return self._detector_list, self._calibration_set

    def _train_single_model(
        self,
        detector: BaseDetector,
        x: pd.DataFrame | np.ndarray,
        bootstrap_indices: np.ndarray,
        seed: int | None,
        iteration: int,
    ) -> BaseDetector:
        """Train a single bootstrap model."""
        model = deepcopy(detector)
        model = set_params(model, seed=seed, random_iteration=True, iteration=iteration)
        model.fit(x[bootstrap_indices])
        return model

    def _train_models_parallel(
        self,
        detector: BaseDetector,
        x: pd.DataFrame | np.ndarray,
        all_bootstrap_indices: np.ndarray,
        seed: int | None,
        n_jobs: int,
        logger,
    ) -> None:
        """Train bootstrap models in parallel."""
        train_func = partial(self._train_single_model, detector, x, seed=seed)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(train_func, all_bootstrap_indices[i], i): i
                for i in range(self._n_bootstraps)
            }

            for future in tqdm(
                as_completed(futures),
                total=self._n_bootstraps,
                desc=f"Parallel bootstrap training ({self._n_bootstraps} iterations)",
                disable=not logger.isEnabledFor(logging.INFO),
            ):
                i = futures[future]
                self._bootstrap_models[i] = future.result()

    def _compute_oob_scores(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute out-of-bag calibration scores for JaB+ method using vectorization.

        This optimized version:
        1. Uses pre-computed boolean mask for OOB membership
        2. Batches predictions for each model across all its OOB samples
        3. Uses vectorized aggregation operations

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

        # Initialize prediction accumulator and count arrays
        prediction_sum = np.zeros(n_samples)
        prediction_count = np.zeros(n_samples, dtype=int)

        # For median calculation, we need to store all predictions
        if self._aggregation_method == Aggregation.MEDIAN:
            all_predictions = [[] for _ in range(n_samples)]

        # Process each bootstrap model
        for model_idx, model in enumerate(self._bootstrap_models):
            # Get OOB samples for this model
            oob_samples = self._oob_mask[model_idx]
            oob_indices = np.where(oob_samples)[0]

            if len(oob_indices) > 0:
                # Batch predict all OOB samples for this model
                oob_predictions = model.decision_function(x[oob_indices])

                if self._aggregation_method == Aggregation.MEAN:
                    # Accumulate for mean calculation
                    prediction_sum[oob_indices] += oob_predictions
                    prediction_count[oob_indices] += 1
                else:
                    # Store for median calculation
                    for idx, pred in zip(oob_indices, oob_predictions):
                        all_predictions[idx].append(pred)

        # Check for samples with no OOB predictions
        if self._aggregation_method == Aggregation.MEAN:
            no_predictions = prediction_count == 0
        else:
            no_predictions = np.array([len(preds) == 0 for preds in all_predictions])

        if np.any(no_predictions):
            raise ValueError(
                f"Samples {np.where(no_predictions)[0]} have no OOB predictions. "
                "Consider increasing n_bootstraps."
            )

        # Compute final scores
        if self._aggregation_method == Aggregation.MEAN:
            oob_scores = prediction_sum / prediction_count
        else:
            oob_scores = np.array([np.median(preds) for preds in all_predictions])

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
