import logging
import math
from collections.abc import Callable
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.logging import get_logger
from nonconform.utils.func.params import set_params
from pyod.models.base import BaseDetector


class Bootstrap(BaseStrategy):
    """Implements bootstrap-based conformal anomaly detection.

    This strategy uses bootstrap resampling to create multiple training sets
    and calibration sets. For each bootstrap iteration:
    1. A random subset of the data is sampled with replacement for training
    2. The remaining samples are used for calibration
    3. Optionally, a fixed number of calibration samples can be selected

    The strategy can operate in two modes:
    1. Standard mode: Uses a single model trained on all data for prediction
    2. Plus mode: Uses an ensemble of models, each trained on a bootstrap sample

    Attributes
    ----------
        _resampling_ratio (float): Proportion of data to use for training in each
            bootstrap iteration
        _n_bootstraps (int): Number of bootstrap iterations
        _n_calib (int | None): Optional fixed number of calibration samples to use
        _plus (bool): Whether to use the plus variant (ensemble of models)
        _detector_list (list[BaseDetector]): List of trained detectors
        _calibration_set (list[float]): List of calibration scores
        _calibration_ids (list[int]): Indices of samples used for calibration
    """

    def __init__(
        self,
        resampling_ratio: float | None = None,
        n_bootstraps: int | None = None,
        n_calib: int | None = None,
        plus: bool = False,
    ):
        """Initialize the Bootstrap strategy.

        Exactly two of `resampling_ratio`, `n_bootstraps`, and `n_calib`
        should be provided. The third will be calculated by `_configure`.

        Args:
            resampling_ratio (Optional[float], optional): The proportion of
                data to use for training in each bootstrap. Defaults to ``None``.
            n_bootstraps (Optional[int], optional): The number of bootstrap
                iterations. Defaults to ``None``.
            n_calib (Optional[int], optional): The desired size of the final
                calibration set. If set, collected scores/IDs might be
                subsampled. Defaults to ``None``.
            plus (bool, optional): If ``True``, appends each bootstrapped model
                to `_detector_list`. If ``False``, `_detector_list` will contain
                one model trained on all data after calibration scores are
                collected. Defaults to ``False``.
        """
        super().__init__(plus)
        self._resampling_ratio: float | None = resampling_ratio
        self._n_bootstraps: int | None = n_bootstraps
        self._n_calib: int | None = n_calib
        self._plus: bool = plus

        self._detector_list: list[BaseDetector] = []
        self._calibration_set: list[float] = []
        self._calibration_ids: list[int] = []

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int | None = None,
        weighted: bool = False,
        iteration_callback: Callable[[int, np.ndarray], None] | None = None,
    ) -> tuple[list[BaseDetector], list[float]]:
        """Fit and calibrate the detector using bootstrap resampling.

        This method implements the bootstrap strategy by:
        1. Creating multiple bootstrap samples of the data
        2. For each bootstrap iteration:
           - Train the detector on the bootstrap sample
           - Use the out-of-bootstrap samples for calibration
           - Store calibration scores and optionally the trained model
        3. If not in plus mode, train a final model on all data
        4. Optionally subsample the calibration set to a fixed size

        The method provides robust calibration scores by using multiple
        bootstrap iterations, which helps account for the variability in
        the data and model training.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Input data matrix of shape
                (n_samples, n_features).
            detector (BaseDetector): The base anomaly detector to be used.
            weighted (bool, optional): Whether to use weighted calibration.
                If True, calibration scores are weighted by their sample
                indices. Defaults to False.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            iteration_callback (Callable[[int, np.ndarray], None], optional):
                Optional callback function that gets called after each bootstrap
                iteration with the iteration number and calibration scores.
                Defaults to None.

        Returns
        -------
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * List of trained detectors (either n_bootstraps models in plus
                  mode or a single model in standard mode)
                * List of calibration scores from all bootstrap iterations

        Raises
        ------
            ValueError: If resampling_ratio is not between 0 and 1, or if
                n_bootstraps is less than 1, or if n_calib is less than 1
                when specified.
        """
        self._configure(len(x))

        _detector = detector
        _generator = np.random.default_rng(seed)

        folds = ShuffleSplit(
            n_splits=self._n_bootstraps,
            train_size=self._resampling_ratio,
            random_state=seed,
        )

        n_folds = folds.get_n_splits()
        last_iteration_index = (
            0  # To ensure unique iteration for final model if not _plus
        )
        logger = get_logger("strategy.bootstrap")
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(
                folds.split(x),
                total=n_folds,
                desc=f"Bootstrap training ({n_folds} folds)",
                disable=not logger.isEnabledFor(logging.INFO),
            )
        ):
            last_iteration_index = i
            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(_detector)
            model = set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[train_idx])

            current_scores = model.decision_function(x[calib_idx])

            # Call iteration callback if provided
            if iteration_callback is not None:
                iteration_callback(i, current_scores)

            if self._plus:
                self._detector_list.append(deepcopy(model))
            self._calibration_set.extend(current_scores)

        if not self._plus:
            model = copy(_detector)
            model = set_params(
                model,
                seed=seed,
                random_iteration=True,
                iteration=(last_iteration_index + 1),
            )
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        if self._n_calib is not None and self._n_calib < len(self._calibration_set):
            ids = _generator.choice(
                len(self._calibration_set), size=self._n_calib, replace=False
            )
            self._calibration_set = [self._calibration_set[i] for i in ids]
            if weighted:
                self._calibration_ids = [self._calibration_ids[i] for i in ids]

        return self._detector_list, self._calibration_set

    def _sanity_check(self) -> None:
        """Ensure that exactly two configuration parameters are provided.

        Verifies that exactly two of `_resampling_ratio`, `_n_bootstraps`,
        and `_n_calib` have been set during initialization. The third
        parameter is derived from these two and the dataset size by
        `_configure`.

        Raises
        ------
            ValueError: If not exactly two of the three parameters
                (resampling_ratio, n_bootstraps, n_calib) are defined.
        """
        num_defined: int = sum(
            param is not None
            for param in (self._resampling_ratio, self._n_bootstraps, self._n_calib)
        )
        if num_defined != 2:
            raise ValueError(
                "Exactly two parameters (resampling_ratio, n_bootstraps, n_calib) "
                "must be defined."
            )

    @staticmethod
    def _calculate_n_calib_target(
        n_data: int, num_bootstraps: int, res_ratio: float
    ) -> int:
        """Calculate the target number of calibration samples.

        Args:
            n_data (int): Total number of data points.
            num_bootstraps (int): Number of bootstrap iterations.
            res_ratio (float): Resampling ratio for training.

        Returns
        -------
            int: Target number of calibration samples.

        Raises
        ------
            ValueError: If resampling ratio is not between 0 and 1, or if
                number of bootstraps is less than 1.
        """
        if not (0 < res_ratio < 1):
            raise ValueError("Resampling ratio must be between 0 and 1.")
        if num_bootstraps < 1:
            raise ValueError("Number of bootstraps must be at least 1.")
        return math.ceil(num_bootstraps * n_data * (1.0 - res_ratio))

    @staticmethod
    def _calculate_n_bootstraps_target(
        n_data: int, num_calib_target: int, res_ratio: float
    ) -> int:
        if not (0 < res_ratio < 1):
            raise ValueError("Resampling ratio must be between 0 and 1.")
        if n_data * (1.0 - res_ratio) <= 0:
            raise ValueError(
                "Product n_data * (1 - res_ratio) must be positive for "
                "calculating n_bootstraps."
            )
        n_b = math.ceil(num_calib_target / (n_data * (1.0 - res_ratio)))
        if n_b < 1:
            raise ValueError("Calculated number of bootstraps is less than 1.")
        return n_b

    @staticmethod
    def _calculate_resampling_ratio_target(
        n_data: int, num_bootstraps: int, num_calib_target: int
    ) -> float:
        if num_bootstraps < 1:
            raise ValueError("Number of bootstraps must be at least 1.")
        if n_data <= 0 or num_bootstraps * n_data == 0:
            raise ValueError("Product n_data * num_bootstraps must be positive.")

        val = 1.0 - (float(num_calib_target) / (num_bootstraps * n_data))
        if not (0 < val < 1):
            raise ValueError(
                f"Calculated resampling_ratio ({val:.3f}) is not between 0 and 1. "
                "Check input n_calib, n_bootstraps, and data size."
            )
        return val

    def _configure(self, n: int) -> None:
        """Configure bootstrap parameters based on two provided settings.

        Calculates and sets the third missing parameter among
        `_resampling_ratio`, `_n_bootstraps`, and `_n_calib` based on the
        two that were provided during initialization and the total number of
        samples `n`. This method modifies the instance attributes in place.

        It calls `_sanity_check` first to ensure valid initial parameters.
        The formulas assume `_n_calib` refers to the total number of unique
        calibration points desired or expected after all bootstrap samples.

        Args:
            n (int): The total number of samples in the dataset.

        Raises
        ------
            ValueError: If `_sanity_check` fails (i.e., not exactly two
                parameters were initially defined), or if calculated
                `resampling_ratio` is not within (0, 1) or `n_bootstraps` < 1.
        """
        self._sanity_check()

        if self._n_bootstraps is not None and self._resampling_ratio is not None:
            self._n_calib = self._calculate_n_calib_target(
                n_data=n,
                num_bootstraps=self._n_bootstraps,
                res_ratio=self._resampling_ratio,
            )
        elif self._n_bootstraps is not None and self._n_calib is not None:
            self._resampling_ratio = self._calculate_resampling_ratio_target(
                n_data=n,
                num_bootstraps=self._n_bootstraps,
                num_calib_target=self._n_calib,
            )
        elif self._resampling_ratio is not None and self._n_calib is not None:
            self._n_bootstraps = self._calculate_n_bootstraps_target(
                n_data=n,
                res_ratio=self._resampling_ratio,
                num_calib_target=self._n_calib,
            )

        # Log configuration information
        logger = get_logger("strategy.bootstrap")
        training_samples_per_iter = int(n * self._resampling_ratio)
        calib_samples_per_iter = n - training_samples_per_iter

        logger.info(
            "Bootstrap Configuration:\n"
            f"  • Data: {n:,} total samples\n"
            f"  • Training: {training_samples_per_iter:,} samples per iteration "
            f"({self._resampling_ratio:.2%} ratio)\n"
            f"  • Calibration: ~{calib_samples_per_iter:,} samples per iteration "
            f"→ {self._n_calib:,} total expected\n"
            f"  • Bootstrap iterations: {self._n_bootstraps:,}"
        )

    @property
    def calibration_ids(self) -> list[int]:
        """Returns the list of indices used for calibration.

        These are indices relative to the original input data `x` provided to
        :meth:`fit_calibrate`. The list contains indices of all out-of-bag
        samples encountered during bootstrap iterations. If `_n_calib` was
        set and `weighted` was ``True`` in `fit_calibrate`, this list might
        be a subsample of all encountered IDs, corresponding to the
        subsampled `_calibration_set`.

        Returns
        -------
            List[int]: A list of integer indices.
        """
        return self._calibration_ids
