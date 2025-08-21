import logging
from collections.abc import Callable
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.enums import Distribution
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import set_params
from pyod.models.base import BaseDetector


class Randomized(BaseStrategy):
    """Implements randomized leave-p-out (rLpO) conformal anomaly detection.

    This strategy uses randomized leave-p-out resampling where on each iteration
    a validation set size p is drawn at random, then a size-p validation set is
    sampled without replacement, the detector is trained on the rest, and
    calibration scores are computed. This approach smoothly interpolates between
    leave-one-out (p=1) and larger holdout strategies.

    The strategy can operate in two modes:
    1. Standard mode: Uses a single model trained on all data for prediction
    2. Plus mode: Uses an ensemble of models, each trained on a different subset

    Attributes
    ----------
        _sampling_distr (Distribution): Distribution type for drawing holdout sizes
        _n_iterations (int | None): Number of rLpO iterations
        _holdout_size_range (tuple): Range of holdout sizes (relative or absolute)
        _beta_params (tuple): Alpha and beta parameters for beta distribution
        _grid_probs (tuple): Holdout sizes and probabilities for grid distribution
        _n_calib (int | None): Target number of calibration samples
        _use_n_calib_mode (bool): Whether to use n_calib mode vs n_iterations mode
        _plus (bool): Whether to use the plus variant (ensemble of models)
        _detector_list (list[BaseDetector]): List of trained detectors
        _calibration_set (list[float]): List of calibration scores
        _calibration_ids (list[int]): Indices of samples used for calibration
    """

    def __init__(
        self,
        n_iterations: int | None = None,
        n_calib: int | None = None,
        sampling_distr: Distribution = Distribution.UNIFORM,
        holdout_size_range: tuple[float, float] | None = None,
        beta_params: tuple[float, float] | None = None,
        grid_probs: tuple[list[int], list[float]] | None = None,
        plus: bool = True,
    ):
        """Initialize the RandomizedLeaveOut strategy.

        Args:
            n_iterations (int | None, optional): Number of rLpO iterations to perform.
                Cannot be used together with n_calib. Defaults to None.
            n_calib (int | None, optional): Target number of calibration samples.
                Iterations will stop when this target is reached or exceeded, then
                subsample to exactly this size. Cannot be used with n_iterations.
                Defaults to None.
            sampling_distr (Distribution, optional): Distribution for drawing holdout
                set sizes. Options: Distribution.BETA_BINOMIAL, Distribution.UNIFORM,
                Distribution.GRID. Defaults to Distribution.UNIFORM.
            holdout_size_range (tuple[float, float], optional): Min and max holdout
                set sizes. Values in ]0, 1[ are interpreted as fractions of dataset
                size. Values >= 1 are interpreted as absolute sample counts.
                If None, defaults to (0.1, 0.5) for relative sizing. Defaults to None.
            beta_params (tuple[float, float], optional): Alpha and beta parameters
                for Beta distribution used to draw holdout size fractions. If None and
                sampling_distr is BETA_BINOMIAL, defaults to (2.0, 5.0).
                Common parameterizations:
                - (1.0, 1.0): Uniform sampling (equivalent to UNIFORM distribution)
                - (2.0, 5.0): Right-skewed, favors smaller holdout sizes [DEFAULT]
                - (5.0, 2.0): Left-skewed, favors larger holdout sizes
                - (2.0, 2.0): Bell-shaped, concentrated around middle sizes
                - (0.5, 0.5): U-shaped, concentrated at extremes
                Defaults to None.
            grid_probs (tuple[list[int], list[float]], optional): Holdout sizes and
                corresponding probabilities for grid distribution. Required if
                sampling_distr is Distribution.GRID. Defaults to None.
            plus (bool, optional): If True, uses ensemble of models trained on
                different subsets. If False, uses single model trained on all data.
                Defaults to True.

        Raises
        ------
            ValueError: If required parameters for the chosen distribution are missing,
                if both n_iterations and n_calib are specified, or neither.
        """
        super().__init__(plus)

        # Validate that exactly one of n_iterations or n_calib is specified
        if n_iterations is not None and n_calib is not None:
            logger = get_logger("strategy.randomized")
            logger.warning(
                "Both n_iterations and n_calib specified. "
                "Using n_calib and ignoring n_iterations."
            )
            n_iterations = None
        elif n_iterations is None and n_calib is None:
            raise ValueError("Must specify either n_iterations or n_calib")

        if n_iterations is not None and n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if n_calib is not None and n_calib < 1:
            raise ValueError("n_calib must be at least 1")

        self._n_iterations: int | None = n_iterations
        self._sampling_distr: Distribution = sampling_distr
        self._holdout_size_range: tuple[float, float] | None = holdout_size_range
        self._beta_params: tuple[float, float] | None = beta_params
        self._grid_probs: tuple[list[int], list[float]] | None = grid_probs
        self._n_calib: int | None = n_calib
        self._plus: bool = plus
        self._use_n_calib_mode: bool = n_calib is not None

        # Warn if plus=False to alert about potential validity issues
        if not plus:
            logger = get_logger("strategy.randomized")
            logger.warning(
                "Setting plus=False may compromise conformal validity. "
                "The plus variant (plus=True) is recommended for validity guarantees."
            )

        # Validate distribution-specific parameters
        self._validate_distribution_params()

        self._detector_list: list[BaseDetector] = []
        self._calibration_set: list[float] = []
        self._calibration_ids: list[int] = []
        self._n_data: int = 0
        self._holdout_sizes: list[int] = []
        self._iteration_scores: list[list[float]] = []
        # Will be set in _configure_holdout_size_range
        self._holdout_size_range_abs: tuple[int, int] = (1, 1)

    def _validate_distribution_params(self) -> None:
        """Validate that required parameters are provided for the chosen distribution.

        Sets default parameters where appropriate and logs when defaults are used.

        Raises
        ------
            ValueError: If required parameters are missing for the distribution.
        """
        logger = get_logger("strategy.randomized")

        if self._sampling_distr == Distribution.BETA_BINOMIAL:
            if self._beta_params is None:
                # Use sensible default: right-skewed, favors smaller holdout sizes
                self._beta_params = (2.0, 5.0)
                logger.info(
                    "Using default beta_params (2.0, 5.0) for BETA_BINOMIAL "
                    "distribution. This creates a right-skewed distribution "
                    "favoring smaller holdout sizes."
                )
        elif self._sampling_distr == Distribution.GRID and self._grid_probs is None:
            raise ValueError("grid_probs required for grid distribution")

        if self._beta_params is not None:
            alpha, beta = self._beta_params
            if alpha <= 0 or beta <= 0:
                raise ValueError("Beta parameters must be positive")

        if self._grid_probs is not None:
            p_values, probabilities = self._grid_probs
            if len(p_values) != len(probabilities):
                raise ValueError("p_values and probabilities must have same length")
            if not np.allclose(sum(probabilities), 1.0):
                raise ValueError("Probabilities must sum to 1.0")
            if any(p <= 0 for p in p_values):
                raise ValueError("All p_values must be positive")

    def _configure_holdout_size_range(self, n: int) -> None:
        """Configure the holdout_size_range based on data size if not provided.

        Args:
            n (int): Total number of data points.
        """
        self._n_data = n
        if self._holdout_size_range is None:
            # Default to relative sizing: 10% to 50% of data
            self._holdout_size_range = (0.1, 0.5)

        # Convert relative values to absolute if needed and validate
        min_size, max_size = self._holdout_size_range

        # Convert relative to absolute
        if min_size < 1.0:
            min_size_abs = max(1, int(min_size * n))
        else:
            min_size_abs = int(min_size)

        if max_size < 1.0:
            max_size_abs = max(1, int(max_size * n))
        else:
            max_size_abs = int(max_size)

        # Validate range
        if min_size_abs < 1 or max_size_abs >= n:
            raise ValueError(
                f"holdout_size_range results in ({min_size_abs}, {max_size_abs}) "
                f"which is invalid for dataset size {n}"
            )
        if min_size_abs > max_size_abs:
            raise ValueError(
                f"holdout_size_range min ({min_size_abs}) > max ({max_size_abs})"
            )

        # Store the absolute range for later use
        self._holdout_size_range_abs = (min_size_abs, max_size_abs)

    def _draw_holdout_size(self, generator: np.random.Generator) -> int:
        """Draw a holdout set size according to the specified distribution.

        Args:
            generator (np.random.Generator): Random number generator.

        Returns
        -------
            int: Holdout set size.
        """
        if self._sampling_distr == Distribution.UNIFORM:
            min_size, max_size = self._holdout_size_range_abs
            return generator.integers(min_size, max_size + 1)

        elif self._sampling_distr == Distribution.BETA_BINOMIAL:
            alpha, beta = self._beta_params
            # Draw from Beta distribution and scale to holdout range
            v = generator.beta(alpha, beta)
            min_size, max_size = self._holdout_size_range_abs
            range_size = max_size - min_size
            size = max(min_size, min(max_size, int(v * range_size + min_size)))
            return size

        elif self._sampling_distr == Distribution.GRID:
            holdout_sizes, probabilities = self._grid_probs
            # Convert relative sizes to absolute if needed
            abs_sizes = []
            for size in holdout_sizes:
                if size < 1.0:
                    abs_sizes.append(max(1, int(size * self._n_data)))
                else:
                    abs_sizes.append(int(size))
            return generator.choice(abs_sizes, p=probabilities)

        else:
            raise ValueError(f"Unknown sampling_distr: {self._sampling_distr}")

    def _log_configuration(self) -> None:
        """Log configuration information at initialization."""
        logger = get_logger("strategy.randomized")
        mode_str = "n_calib mode" if self._use_n_calib_mode else "n_iterations mode"

        logger.info(
            f"RandomizedLeaveOut Configuration ({mode_str}):\n"
            f"  Data: {self._n_data:,} total samples\n"
            f"  Distribution: {self._sampling_distr}\n"
            f"  Holdout size range: {self._holdout_size_range}\n"
            f"  Plus mode: {self._plus}"
        )

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int | None = None,
        weighted: bool = False,
        iteration_callback: Callable[[int, np.ndarray], None] | None = None,
        track_p_values: bool = False,
    ) -> tuple[list[BaseDetector], list[float]]:
        """Fit and calibrate the detector using randomized leave-p-out resampling.

        This method implements the rLpO strategy by:
        1. For each iteration, drawing a random holdout set size
        2. Sampling a holdout set of that size without replacement
        3. Training the detector on the remaining samples
        4. Computing calibration scores on the holdout set
        5. Optionally storing the trained model (in plus mode)
        6. If using n_calib mode, stopping when target calibration size is reached

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Input data matrix of shape
                (n_samples, n_features).
            detector (BaseDetector): The base anomaly detector to be used.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            weighted (bool, optional): Whether to store calibration sample indices.
                Defaults to False.
            iteration_callback (Callable[[int, np.ndarray], None], optional):
                Optional callback function called after each iteration with the
                iteration number and calibration scores. Defaults to None.
            track_p_values (bool, optional): If True, stores the holdout sizes and
                per-iteration scores for performance analysis. Can be accessed
                via get_iteration_info(). Defaults to False.

        Returns
        -------
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * List of trained detectors (either multiple models in plus
                  mode or a single model in standard mode)
                * List of calibration scores from all iterations

        Raises
        ------
            ValueError: If holdout set size would leave insufficient training data.
        """
        self._configure_holdout_size_range(len(x))
        self._log_configuration()

        _detector = detector
        generator = np.random.default_rng(seed)

        logger = get_logger("strategy.randomized")

        # Determine iteration strategy and progress bar setup
        if self._use_n_calib_mode:
            # Use a high iteration limit but stop when n_calib is reached
            max_iterations = 10000  # Reasonable upper bound
            base_desc = f"Randomized (target: {self._n_calib})"
            total_for_progress = self._n_calib
        else:
            max_iterations = self._n_iterations
            base_desc = f"Randomized ({self._n_iterations} iterations)"
            total_for_progress = self._n_iterations

        actual_iterations = 0
        running_holdout_sum = 0
        with tqdm(
            total=total_for_progress,
            desc=base_desc,
            disable=not logger.isEnabledFor(logging.INFO),
        ) as pbar:
            while True:
                # Check stopping condition
                if self._use_n_calib_mode:
                    if len(self._calibration_set) >= self._n_calib:
                        break
                    if actual_iterations >= max_iterations:
                        logger.warning(
                            f"Reached maximum iterations ({max_iterations}) "
                            f"with only {len(self._calibration_set)} samples. "
                            f"Target was {self._n_calib}."
                        )
                        break
                else:
                    if actual_iterations >= self._n_iterations:
                        break

                # Draw holdout set size for this iteration
                holdout_size = self._draw_holdout_size(generator)

                # Sample holdout set without replacement
                all_indices = np.arange(self._n_data)
                calib_idx = generator.choice(
                    all_indices, size=holdout_size, replace=False
                )
                train_idx = np.setdiff1d(all_indices, calib_idx)

                if len(train_idx) < 1:
                    raise ValueError(
                        f"No training samples left with holdout_size={holdout_size} "
                        f"for n={self._n_data}"
                    )

                # Store calibration indices
                self._calibration_ids.extend(calib_idx.tolist())

                # Train model on training set
                model = copy(_detector)
                model = set_params(
                    model, seed=seed, random_iteration=True, iteration=actual_iterations
                )
                model.fit(x[train_idx])

                # Compute calibration scores on holdout set
                current_scores = model.decision_function(x[calib_idx])

                # Call iteration callback if provided
                if iteration_callback is not None:
                    iteration_callback(actual_iterations, current_scores)

                # Store model if in plus mode
                if self._plus:
                    self._detector_list.append(deepcopy(model))

                # Store calibration scores
                self._calibration_set.extend(current_scores.tolist())

                # Track holdout sizes and per-iteration scores if requested
                if track_p_values:
                    self._holdout_sizes.append(holdout_size)
                    self._iteration_scores.append(current_scores.tolist())

                actual_iterations += 1
                running_holdout_sum += holdout_size
                avg_holdout = running_holdout_sum / actual_iterations

                # Update progress bar based on mode
                if self._use_n_calib_mode:
                    # Update progress to show current calibration samples
                    pbar.n = min(len(self._calibration_set), self._n_calib)
                    pbar.desc = (
                        f"{base_desc} | iter: {actual_iterations}, "
                        f"avg_holdout: {avg_holdout:.1f}"
                    )
                    pbar.refresh()
                else:
                    pbar.update(1)
                    pbar.desc = f"{base_desc} | avg_holdout: {avg_holdout:.1f}"

        # If not in plus mode, train final model on all data
        if not self._plus:
            final_model = copy(_detector)
            final_model = set_params(
                final_model,
                seed=seed,
                random_iteration=True,
                iteration=actual_iterations,
            )
            final_model.fit(x)
            self._detector_list.append(deepcopy(final_model))

        # Always subsample to exact n_calib in n_calib mode
        if self._use_n_calib_mode and len(self._calibration_set) != self._n_calib:
            generator = np.random.default_rng(seed)
            if len(self._calibration_set) > self._n_calib:
                # Subsample to exact target
                ids = generator.choice(
                    len(self._calibration_set), size=self._n_calib, replace=False
                )
            else:
                # We have fewer than target - use all available
                ids = list(range(len(self._calibration_set)))
                logger.warning(
                    f"Only collected {len(self._calibration_set)} calibration samples, "
                    f"less than target {self._n_calib}"
                )

            self._calibration_set = [self._calibration_set[i] for i in ids]
            if weighted:
                self._calibration_ids = [self._calibration_ids[i] for i in ids]

            # Also subsample tracking data if enabled
            if track_p_values and self._holdout_sizes:
                # For simplicity, subsample proportionally across iterations
                selected_iterations = (
                    set(ids % actual_iterations) if actual_iterations > 0 else set()
                )
                self._holdout_sizes = [
                    size
                    for i, size in enumerate(self._holdout_sizes)
                    if i in selected_iterations
                ]
                self._iteration_scores = [
                    scores
                    for i, scores in enumerate(self._iteration_scores)
                    if i in selected_iterations
                ]

        # Log final results - only for n_iterations mode
        if not self._use_n_calib_mode:
            final_calib_size = len(self._calibration_set)
            logger.info(f"Final calibration scores: {final_calib_size:,}")

        return self._detector_list, self._calibration_set

    def get_iteration_info(self) -> tuple[list[int], list[list[float]]] | None:
        """Get holdout sizes and per-iteration scores if tracking was enabled.

        This method provides access to the holdout set sizes used in each
        iteration and the corresponding anomaly scores. This information can be
        used for performance analysis, plotting vs. holdout size, or understanding
        the distribution of holdout set sizes used.

        Returns
        -------
            tuple[list[int], list[list[float]]] | None: A tuple containing:
                * List of holdout sizes for each iteration
                * List of score arrays, one per iteration
                Returns None if track_p_values was False during fit_calibrate.
        Example
        -------
            >>> from nonconform.utils.func.enums import Distribution
            >>> strategy = Randomized(n_calib=1000)
            >>> strategy.fit_calibrate(X, detector, track_p_values=True)
            >>> holdout_sizes, scores = strategy.get_iteration_info()
            >>> # holdout_sizes[i] is the holdout set size for iteration i
            >>> # scores[i] are the anomaly scores for iteration i
        """
        if not self._holdout_sizes:  # Empty list means tracking was not enabled
            return None
        return (
            self._holdout_sizes.copy(),
            [scores.copy() for scores in self._iteration_scores],
        )

    @property
    def calibration_ids(self) -> list[int]:
        """Returns the list of indices used for calibration.

        These are indices relative to the original input data `x` provided to
        :meth:`fit_calibrate`. The list contains indices of all holdout
        samples encountered during rLpO iterations.

        Returns
        -------
            list[int]: A list of integer indices for calibration samples.
        """
        return self._calibration_ids
