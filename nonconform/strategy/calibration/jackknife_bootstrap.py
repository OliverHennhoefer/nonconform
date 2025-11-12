import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.strategy.calibration.base import BaseStrategy
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import _set_params
from pyod.models.base import BaseDetector


class JackknifeBootstrap(BaseStrategy):
    """Implements Jackknife+-after-Bootstrap (JaB+) conformal anomaly detection.

    This strategy implements the JaB+ method which provides predictive inference
    for ensemble models trained on bootstrap samples. The key insight is that
    JaB+ uses the out-of-bag (OOB) samples from bootstrap iterations to compute
    calibration scores without requiring additional model training.

    The strategy can operate in two modes:
    1. Plus mode (plus=True): Uses ensemble of models for prediction (recommended)
    2. Standard mode (plus=False): Uses single model trained on all data

    Attributes:
        _n_bootstraps (int): Number of bootstrap iterations
        _aggregation_method (Aggregation): How to aggregate OOB predictions
        _plus (bool): Whether to use the plus variant (ensemble of models)
        _detector_list (list[BaseDetector]): List of trained detectors (ensemble/single)
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
        plus: bool = True,
    ):
        """Initialize the Bootstrap (JaB+) strategy.

        Args:
            n_bootstraps (int, optional): Number of bootstrap iterations.
                Defaults to 100.
            aggregation_method (Aggregation, optional): Method to aggregate out-of-bag
                predictions. Options are Aggregation.MEAN or Aggregation.MEDIAN.
                Defaults to Aggregation.MEAN.
            plus (bool, optional): If True, uses ensemble of bootstrap models for
                prediction (maintains statistical validity). If False, uses single
                model trained on all data. Strongly recommended to use True.
                Defaults to True.

        Raises:
            ValueError: If aggregation_method is not a valid Aggregation enum value.
            ValueError: If n_bootstraps is less than 1.
        """
        super().__init__(plus=plus)

        if n_bootstraps < 2:
            exc = ValueError(
                f"Number of bootstraps must be at least 2, got {n_bootstraps}. "
                f"Typical values are 50-200 for jackknife-after-bootstrap."
            )
            exc.add_note(f"Received n_bootstraps={n_bootstraps}, which is invalid.")
            exc.add_note(
                "Jackknife-after-Bootstrap requires at least two bootstrap iterations "
                "to guarantee that every sample appears out-of-bag."
            )
            exc.add_note("Consider using n_bootstraps=100 as a balanced default.")
            raise exc
        if aggregation_method not in [Aggregation.MEAN, Aggregation.MEDIAN]:
            exc = ValueError(
                f"aggregation_method must be Aggregation.MEAN or Aggregation.MEDIAN, "
                f"got {aggregation_method}. These are the only statistically valid "
                f"methods for combining out-of-bag predictions in JackknifeBootstrap()."
            )
            exc.add_note(f"Received aggregation_method={aggregation_method}")
            exc.add_note("Valid options are: Aggregation.MEAN, Aggregation.MEDIAN")
            exc.add_note(
                "These methods ensure statistical validity of the JaB+ procedure."
            )
            raise exc

        # Warn if plus=False to alert about potential validity issues
        if not plus:
            logger = get_logger("strategy.jackknife_bootstrap")
            logger.warning(
                "Setting plus=False may compromise conformal validity. "
                "The plus variant (plus=True) is recommended "
                "for statistical guarantees."
            )

        self._n_bootstraps: int = n_bootstraps
        self._aggregation_method: Aggregation = aggregation_method

        self._detector_list: list[BaseDetector] = []
        self._calibration_set: np.ndarray = np.array([])
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
    ) -> tuple[list[BaseDetector], np.ndarray]:
        """Fit and calibrate using Jackknife+-after-Bootstrap method.

        This method implements the JaB+ algorithm:
        1. Generate bootstrap samples and train models
        2. For each sample, compute out-of-bag predictions
        3. Aggregate OOB predictions to get calibration scores
        4. Train final model on all data

        Args:
            x (pd.DataFrame | np.ndarray): Input data matrix of shape
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

        Returns:
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * List of trained detector models (if plus=True, single if plus=False)
                * Array of calibration scores from JaB+ procedure
        """
        n_samples = len(x)
        logger = get_logger("strategy.bootstrap")
        generator = np.random.default_rng(seed)

        logger.info(
            f"Bootstrap (JaB+) Configuration:\n"
            f"  - Data: {n_samples:,} total samples\n"
            f"  - Bootstrap iterations: {self._n_bootstraps:,}\n"
            f"  - Aggregation method: {self._aggregation_method}"
        )

        # Step 1: Pre-allocate data structures and generate bootstrap samples
        self._bootstrap_models = [None] * self._n_bootstraps
        all_bootstrap_indices, self._oob_mask = self._generate_bootstrap_indices(
            generator, n_samples
        )

        # Train models (with optional parallelization)
        if n_jobs is None or n_jobs == 1:
            # Sequential training
            bootstrap_iterator = (
                tqdm(
                    range(self._n_bootstraps),
                    desc="Calibration",
                )
                if logger.isEnabledFor(logging.INFO)
                else range(self._n_bootstraps)
            )
            for i in bootstrap_iterator:
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

        self._calibration_set = oob_scores
        self._calibration_ids = list(range(n_samples))

        # Step 3: Handle plus variant
        if self._plus:
            # Plus variant: Use ensemble of bootstrap models for prediction
            self._detector_list = self._bootstrap_models.copy()
            logger.info(
                f"JaB+ calibration completed with {len(self._calibration_set)} scores "
                f"using ensemble of {len(self._bootstrap_models)} models"
            )
        else:
            # Standard variant: Train final model on all data
            final_model = deepcopy(detector)
            final_model = _set_params(
                final_model,
                seed=seed,
                random_iteration=True,
                iteration=self._n_bootstraps,
            )
            final_model.fit(x)
            self._detector_list = [final_model]
            logger.info(
                f"JaB+ calibration completed with {len(self._calibration_set)} scores "
                f"using single model trained on all data"
            )

        return self._detector_list, self._calibration_set

    def _generate_bootstrap_indices(
        self, generator: np.random.Generator, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate bootstrap indices while guaranteeing OOB coverage."""
        if n_samples < 2:
            raise ValueError(
                "JackknifeBootstrap requires at least two samples to form "
                "out-of-bag predictions."
            )
        if self._n_bootstraps < 2:
            raise ValueError(
                "n_bootstraps must be at least 2 to ensure every sample has "
                "an out-of-bag prediction."
            )

        indices = np.empty((self._n_bootstraps, n_samples), dtype=int)
        oob_mask = np.zeros((self._n_bootstraps, n_samples), dtype=bool)
        coverage = np.zeros(n_samples, dtype=bool)
        population = np.arange(n_samples)

        for i in range(self._n_bootstraps):
            uncovered = np.where(~coverage)[0]
            if uncovered.size == 0:
                draw_pool = population
            else:
                shuffled_uncovered = generator.permutation(uncovered)
                remaining_iters = self._n_bootstraps - i
                chunk_size = int(np.ceil(shuffled_uncovered.size / remaining_iters))
                chunk_size = min(chunk_size, n_samples - 1)
                chunk_size = max(1, chunk_size)
                chunk = shuffled_uncovered[:chunk_size]
                draw_mask = np.ones(n_samples, dtype=bool)
                draw_mask[chunk] = False
                draw_pool = population[draw_mask]

            indices[i] = generator.choice(draw_pool, size=n_samples, replace=True)
            in_bag_mask = np.zeros(n_samples, dtype=bool)
            in_bag_mask[indices[i]] = True
            oob_mask[i] = ~in_bag_mask
            coverage |= oob_mask[i]

        uncovered = np.where(~coverage)[0]
        if uncovered.size > 0:
            raise ValueError(
                "Failed to generate complete OOB coverage. "
                "Consider increasing n_bootstraps."
            )
        return indices, oob_mask

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
        model = _set_params(
            model, seed=seed, random_iteration=True, iteration=iteration
        )
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

            future_iterator = (
                tqdm(
                    as_completed(futures),
                    total=self._n_bootstraps,
                    desc="Calibration",
                )
                if logger.isEnabledFor(logging.INFO)
                else as_completed(futures)
            )
            for future in future_iterator:
                i = futures[future]
                self._bootstrap_models[i] = future.result()

    def _aggregate_predictions(self, predictions: list | np.ndarray) -> float:
        """Aggregate predictions using the configured method.

        This centralizes the aggregation logic to follow DRY principle.

        Args:
            predictions: List or array of predictions to aggregate

        Returns:
            float: Aggregated value (mean or median)
        """
        if len(predictions) == 0:
            return np.nan

        match self._aggregation_method:
            case Aggregation.MEAN:
                return np.mean(predictions)
            case Aggregation.MEDIAN:
                return np.median(predictions)
            case _:
                # Should not happen due to validation in __init__
                raise ValueError(
                    f"Unsupported aggregation method: {self._aggregation_method}"
                )

    def _compute_oob_scores(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute out-of-bag calibration scores using centralized aggregation.

        This version eliminates code duplication by using a single helper method
        for all aggregation logic, following the DRY principle.

        Args:
            x (pd.DataFrame | np.ndarray): Input data matrix.

        Returns:
            np.ndarray: Array of calibration scores for each sample.

        Raises:
            ValueError: If a sample has no out-of-bag predictions (very unlikely).
        """
        n_samples = len(x)

        # Collect all predictions per sample
        all_predictions = [[] for _ in range(n_samples)]

        # Process each bootstrap model
        for model_idx, model in enumerate(self._bootstrap_models):
            oob_samples = self._oob_mask[model_idx]
            oob_indices = np.where(oob_samples)[0]

            if len(oob_indices) > 0:
                oob_predictions = model.decision_function(x[oob_indices])
                for idx, pred in zip(oob_indices, oob_predictions):
                    all_predictions[idx].append(pred)

        # Check for samples with no OOB predictions
        no_predictions = np.array([len(preds) == 0 for preds in all_predictions])
        if np.any(no_predictions):
            raise ValueError(
                f"Samples {np.where(no_predictions)[0]} have no OOB predictions. "
                "Consider increasing n_bootstraps."
            )

        # Use centralized aggregation logic
        oob_scores = np.array(
            [self._aggregate_predictions(preds) for preds in all_predictions]
        )
        return oob_scores

    @property
    def calibration_ids(self) -> list[int]:
        """Returns a copy of the list of indices used for calibration.

        In JaB+, all original training samples contribute to calibration
        through the out-of-bag mechanism.

        Returns:
            list[int]: Copy of integer indices (0 to n_samples-1).

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._calibration_ids.copy()

    @property
    def n_bootstraps(self) -> int:
        """Returns the number of bootstrap iterations."""
        return self._n_bootstraps

    @property
    def aggregation_method(self) -> Aggregation:
        """Returns the aggregation method used for OOB predictions."""
        return self._aggregation_method
