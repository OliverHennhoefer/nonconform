"""Calibration strategies for conformal anomaly detection.

These strategies define how detector replicas and calibration scores are formed.
``Split`` uses disjoint fitting and calibration subsets. ``CrossValidation``
uses out-of-fold scores, and ``JackknifeBootstrap`` uses out-of-bag scores. The
latter two are package-specific score-aggregation constructions; their names do
not by themselves transfer coverage theorems for conformal prediction intervals
to anomaly p-values.

``DerandomizedSplits`` retains separate held-out calibration rows and constructs
e-values per split before uniform evidence aggregation and e-BH selection.

Classes:
    BaseStrategy: Abstract base class for calibration strategies.
    Split: Simple train-test split strategy.
    DerandomizedSplits: Repeated split-conformal e-values with e-BH selection.
    CrossValidation: K-fold cross-validation strategy (includes Jackknife factory).
    JackknifeBootstrap: Bootstrap out-of-bag calibration strategy.
"""

from __future__ import annotations

import abc
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from nonconform._internal import (
    BootstrapAggregationMethod,
    ConformalMode,
    ensure_numpy_array,
    normalize_bootstrap_aggregation_method,
    set_params,
)
from nonconform._internal.validation import (
    as_1d_numeric,
    validate_finite,
    validate_optional_seed,
    validate_positive_integer,
    validate_probability,
)

if TYPE_CHECKING:
    from nonconform.fdr import EValueSelectionResult
    from nonconform.structures import AnomalyDetector

# Module-level loggers for performance
_crossval_logger = logging.getLogger("nonconform.resampling.crossval")
_bootstrap_logger = logging.getLogger("nonconform.resampling.bootstrap")

ConformalModeInput = ConformalMode | Literal["plus", "single_model"]


def _normalize_mode(mode: ConformalModeInput) -> ConformalMode:
    """Normalize user-facing mode inputs into ConformalMode enums."""
    if isinstance(mode, ConformalMode):
        return mode
    if mode == "plus":
        return ConformalMode.PLUS
    if mode == "single_model":
        return ConformalMode.SINGLE_MODEL
    raise ValueError(
        "mode must be one of {'plus', 'single_model'} or ConformalMode. "
        f"Got {mode!r}."
    )


class BaseStrategy(abc.ABC):
    """Abstract base class for anomaly detection calibration strategies.

    This class provides a common interface for various calibration strategies
    applied to anomaly detectors. Subclasses must implement the core
    calibration logic and define how calibration data is identified and used.

    Attributes:
        _mode: Model retention mode controlling calibration/inference behavior.
    """

    _uses_e_values = False

    def _select_e_values(
        self, test_scores: np.ndarray, calib_scores: np.ndarray, *, alpha: float
    ) -> EValueSelectionResult:
        """Select from preserved repetitions for an e-value strategy."""
        raise NotImplementedError("This strategy does not construct e-values.")

    def __init__(self, mode: ConformalModeInput = "plus") -> None:
        """Initialize the base calibration strategy.

        Args:
            mode: Model retention mode (`"plus"` or `"single_model"`).
                Equivalent ``ConformalMode`` enum values are also accepted.
        """
        self._mode: ConformalMode = _normalize_mode(mode)
        self._calibration_ids: list[int] = []

    @abc.abstractmethod
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fits the detector and performs calibration.

        Args:
            x: The input data for fitting and calibration.
            detector: The anomaly detection model to be fitted and calibrated.
            seed: Random seed for reproducibility. Defaults to None.
            weighted: Whether to use weighted approach. Defaults to False.

        Returns:
            Tuple of (list of trained detectors, calibration scores array).
        """
        raise NotImplementedError(
            "The fit_calibrate() method must be implemented by subclasses."
        )

    @property
    @abc.abstractmethod
    def calibration_ids(self) -> list[int] | None:
        """Indices of data points used for calibration."""
        pass


class Split(BaseStrategy):
    """Split conformal strategy for fast anomaly detection.

    Implements the classical split conformal approach by dividing training data
    into separate fitting and calibration sets.

    Args:
        n_calib: Size or proportion of data used for calibration.
            If float, must be between 0.0 and 1.0 (proportion).
            If int, the absolute number of samples. Defaults to 0.1.

    Examples:
        ```python
        from nonconform import Split

        # Use 20% of data for calibration
        strategy = Split(n_calib=0.2)

        # Use exactly 1000 samples for calibration
        strategy = Split(n_calib=1000)
        ```
    """

    def __init__(self, n_calib: float | int = 0.1) -> None:
        super().__init__()
        self._calib_size: float | int = n_calib
        self._calibration_ids: list[int] | None = None

    def _validate_n_calib(self, n_samples: int) -> None:
        """Validate calibration size against dataset size."""
        n_calib = self._calib_size
        if isinstance(n_calib, float):
            if not (0 < n_calib < 1):
                raise ValueError(
                    f"Proportional n_calib must be in (0, 1), got {n_calib}"
                )
            n_calib_abs = math.ceil(n_samples * n_calib)
        elif isinstance(n_calib, int):
            if n_calib < 1:
                raise ValueError(
                    f"Absolute n_calib must be in [1, {n_samples}), got {n_calib}"
                )
            n_calib_abs = n_calib
        else:
            raise TypeError(f"n_calib must be int or float, got {type(n_calib)}")

        if n_calib_abs >= n_samples:
            if n_calib_abs == n_samples:
                raise ValueError(
                    "No training data remaining after calibration split. "
                    "Reduce n_calib to leave data for training the base detector."
                )
            raise ValueError(
                f"Calibration size ({n_calib_abs}) exceeds training size ({n_samples})"
            )

    @ensure_numpy_array
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        weighted: bool = False,
        seed: int | None = None,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fits detector and generates calibration scores using a data split.

        Args:
            x: The input data.
            detector: The detector instance to train.
            weighted: If True, stores calibration sample indices. Defaults to False.
            seed: Random seed for reproducibility. Defaults to None.

        Returns:
            Tuple of (list with trained detector, calibration scores array).
        """
        self._validate_n_calib(len(x))
        x_id = np.arange(len(x))
        train_id, calib_id = train_test_split(
            x_id, test_size=self._calib_size, shuffle=True, random_state=seed
        )

        if hasattr(detector, "set_params"):
            try:
                detector.set_params(random_state=seed)
            except (TypeError, ValueError):
                pass  # Detector may not support random_state parameter

        detector.fit(x[train_id])
        calibration_set = detector.decision_function(x[calib_id])

        if weighted:
            self._calibration_ids = calib_id.tolist()
        else:
            self._calibration_ids = None
        return [detector], calibration_set

    @property
    def calibration_ids(self) -> list[int] | None:
        """Indices of calibration samples (None if weighted=False)."""
        return (
            self._calibration_ids.copy() if self._calibration_ids is not None else None
        )

    @property
    def calib_size(self) -> float | int:
        """Returns the calibration size or proportion."""
        return self._calib_size


class DerandomizedSplits(BaseStrategy):
    """Aggregate conformal e-values across repeated random splits with e-BH.

    Each replica retains its own held-out calibration scores. Selection converts
    each replica's scores to e-values before uniformly averaging the evidence;
    it never pools calibration scores or aggregates raw scores first.

    Args:
        n_repetitions: Positive number of splits. Defaults to five.
        n_calib: Calibration count or fraction, with the same meaning as Split.
        alpha_bh: Fixed inner threshold in (0, 1), or None for alpha / 10 using
            the target supplied to detector.select(). Choose before inspecting
            the test evidence.
        tie_seed: Optional non-negative override for randomized score ties.
            None automatically derives a separate random stream during fit.

    Examples:
        ```python
        from sklearn.ensemble import IsolationForest
        from nonconform import ConformalDetector, DerandomizedSplits

        detector = ConformalDetector(
            detector=IsolationForest(),
            strategy=DerandomizedSplits(n_repetitions=5, n_calib=0.2),
            seed=42,
        )
        # detector.fit(x_reference)
        # selected = detector.select(x_test, alpha=0.05)
        # evidence = detector.last_selection_result.e_values
        ```

    Note:
        Requires unweighted, integrated splits and exchangeable normal reference
        and null test observations. All repetitions score the same fixed test
        family. The aggregate null-evidence condition supports one final e-BH
        application; individual values need not be ordinary e-values. Repetition
        reduces dependence on a particular split but does not remove randomness.
    """

    _uses_e_values = True

    def __init__(
        self,
        n_repetitions: int = 5,
        n_calib: float | int = 0.1,
        *,
        alpha_bh: float | None = None,
        tie_seed: int | None = None,
    ) -> None:
        super().__init__()
        validate_positive_integer("n_repetitions", n_repetitions)
        if alpha_bh is not None:
            validate_probability("alpha_bh", alpha_bh)
        validate_optional_seed("tie_seed", tie_seed)
        self.n_repetitions = n_repetitions
        self.n_calib = n_calib
        self.alpha_bh = alpha_bh
        self.tie_seed = tie_seed
        self._effective_tie_seed: int | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for inspection and sklearn cloning."""
        return {
            "n_repetitions": self.n_repetitions,
            "n_calib": self.n_calib,
            "alpha_bh": self.alpha_bh,
            "tie_seed": self.tie_seed,
        }

    def set_params(self, **params: Any) -> Self:
        """Update constructor parameters and clear derived random state."""
        current = self.get_params()
        unknown = params.keys() - current.keys()
        if unknown:
            raise ValueError(
                f"Invalid DerandomizedSplits parameters: {sorted(unknown)}"
            )
        updated = type(self)(**(current | params))
        self.__dict__.update(updated.__dict__)
        return self

    @ensure_numpy_array
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fit independent replicas and return aligned calibration-score rows.

        Returns:
            Models and scores shaped (n_repetitions, n_calibration). Row i
            contains only held-out scores produced by model i.
        """
        self._effective_tie_seed = None
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError(
                "x must be a two-dimensional reference batch with features."
            )
        if weighted:
            raise ValueError(
                "DerandomizedSplits does not support weighted calibration."
            )
        validate_optional_seed("seed", seed)
        validate_positive_integer("n_repetitions", self.n_repetitions)
        split = Split(n_calib=self.n_calib)
        split._validate_n_calib(len(x))
        n_calibration = (
            math.ceil(len(x) * self.n_calib)
            if isinstance(self.n_calib, float)
            else self.n_calib
        )
        split_stream, tie_stream = np.random.SeedSequence(seed).spawn(2)
        models = []
        calibration_rows = []
        for stream in split_stream.spawn(self.n_repetitions):
            split_seed = int(stream.generate_state(1)[0])
            replica = set_params(deepcopy(detector), split_seed)
            fitted, scores = split.fit_calibrate(x, replica, seed=split_seed)
            row = as_1d_numeric("calibration scores", scores)
            if row.size != n_calibration:
                raise ValueError(
                    "Each model must return one score per calibration row."
                )
            validate_finite("calibration scores", row)
            models.append(fitted[0])
            calibration_rows.append(row.copy())
        calibration_scores = np.vstack(calibration_rows)
        self._effective_tie_seed = (
            int(tie_stream.generate_state(1, dtype=np.uint64)[0])
            if self.tie_seed is None
            else self.tie_seed
        )
        return models, calibration_scores

    def _select_e_values(
        self, test_scores: np.ndarray, calib_scores: np.ndarray, *, alpha: float
    ) -> EValueSelectionResult:
        """Delegate preserved score rows to the shared statistical procedure."""
        from nonconform.fdr import _select_conformal_e_values_from_scores

        if self._effective_tie_seed is None:
            raise RuntimeError("Fit DerandomizedSplits before selecting anomalies.")
        return _select_conformal_e_values_from_scores(
            test_scores,
            calib_scores,
            alpha=alpha,
            alpha_bh=self.alpha_bh,
            tie_seed=self._effective_tie_seed,
        )

    @property
    def calibration_ids(self) -> None:
        """No pooled calibration sample exists for this unweighted strategy."""
        return None


class CrossValidation(BaseStrategy):
    """K-fold out-of-fold calibration for conformal anomaly scoring.

    The strategy trains one detector per fold and records scores for observations
    while they are held out. In ``"plus"`` mode, test scores are aggregated over
    the retained fold models before comparison with the out-of-fold calibration
    scores. This is the package's anomaly-score construction, not a claim that a
    CV+ prediction-interval theorem applies unchanged.

    Args:
        k: Number of folds. If None, uses leave-one-out (k=n at fit time).
        mode: Model retention mode (`"plus"` or `"single_model"`). Equivalent
            ``ConformalMode`` values are accepted. Defaults to `"plus"`.
        shuffle: Whether to shuffle data before splitting. Defaults to True.
            Set to False for deterministic leave-one-out (Jackknife).

    Examples:
        ```python
        from nonconform import CrossValidation

        # 5-fold cross-validation
        strategy = CrossValidation(k=5)

        # Leave-one-out (Jackknife) via factory
        strategy = CrossValidation.jackknife()
        ```
    """

    def __init__(
        self,
        k: int | None = 5,
        mode: ConformalModeInput = "plus",
        shuffle: bool = True,
    ) -> None:
        super().__init__(mode)
        if not isinstance(shuffle, bool):
            raise TypeError(
                f"shuffle must be a boolean value, got {type(shuffle).__name__}."
            )
        self._k: int | None = k
        self._shuffle: bool = shuffle
        self._is_jackknife = k is None

        # Warn if using single-model mode
        if self._mode is ConformalMode.SINGLE_MODEL:
            _crossval_logger.warning(
                "Setting mode=ConformalMode.SINGLE_MODEL may compromise conformal "
                "validity. mode=ConformalMode.PLUS is recommended."
            )

        self._detector_list: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_ids: list[int] = []

    @classmethod
    def jackknife(cls, mode: ConformalModeInput = "plus") -> CrossValidation:
        """Create Leave-One-Out cross-validation (deterministic, no shuffle).

        This factory method creates a Jackknife strategy, which is a special
        case of k-fold CV where k equals n (the dataset size). Each sample is
        left out exactly once for calibration.

        Args:
            mode: Model retention mode (`"plus"` or `"single_model"`).

        Returns:
            CrossValidation configured for leave-one-out.

        Examples:
            ```python
            from nonconform import CrossValidation

            strategy = CrossValidation.jackknife()
            print(strategy.k, strategy.mode)
            ```
        """
        return cls(k=None, mode=mode, shuffle=False)

    @ensure_numpy_array
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fit and calibrate using k-fold cross-validation.

        Args:
            x: Input data matrix.
            detector: The base anomaly detector.
            seed: Random seed for reproducibility. Defaults to None.
            weighted: Whether to use weighted calibration. Defaults to False.

        Returns:
            Tuple of (list of trained detectors, calibration scores array).

        Raises:
            ValueError: If k < 2 or not enough samples for specified k.
        """
        self._detector_list.clear()
        self._calibration_ids = []

        detector_ = detector
        n_samples = len(x)

        # Determine k (for jackknife mode, k=n)
        k = n_samples if self._is_jackknife else self._k

        if k < 2:
            exc = ValueError(
                f"k must be at least 2 for k-fold cross-validation, got {k}"
            )
            exc.add_note(f"Received k={k}, which is invalid.")
            exc.add_note(
                "Cross-validation requires at least one split for training "
                "and one for calibration."
            )
            raise exc

        if n_samples < k:
            exc = ValueError(
                f"Not enough samples ({n_samples}) for "
                f"k-fold cross-validation with k={k}"
            )
            exc.add_note(f"Each fold needs at least 1 sample, but {n_samples} < {k}.")
            raise exc

        self._calibration_set = np.empty(n_samples, dtype=np.float64)
        calibration_offset = 0

        folds = KFold(
            n_splits=k,
            shuffle=self._shuffle,
            random_state=seed if self._shuffle else None,
        )

        fold_iterator = (
            tqdm(folds.split(x), total=k, desc="Calibration")
            if _crossval_logger.isEnabledFor(logging.INFO)
            else folds.split(x)
        )

        for i, (train_idx, calib_idx) in enumerate(fold_iterator):
            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(detector_)
            if hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=seed)
                except (TypeError, ValueError):
                    pass  # Detector may not support random_state parameter
            model.fit(x[train_idx])

            if self._mode is ConformalMode.PLUS:
                self._detector_list.append(deepcopy(model))

            fold_scores = model.decision_function(x[calib_idx])
            n_fold_samples = len(fold_scores)
            end_idx = calibration_offset + n_fold_samples
            self._calibration_set[calibration_offset:end_idx] = fold_scores
            calibration_offset += n_fold_samples

        if self._mode is ConformalMode.SINGLE_MODEL:
            model = copy(detector_)
            if hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=seed)
                except (TypeError, ValueError):
                    pass  # Detector may not support random_state parameter
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self) -> list[int]:
        """Indices of samples used for calibration."""
        return self._calibration_ids.copy()

    @property
    def k(self) -> int | None:
        """Number of folds (None for jackknife mode)."""
        return self._k

    @property
    def mode(self) -> Literal["plus", "single_model"]:
        """User-facing model retention mode."""
        return "plus" if self._mode is ConformalMode.PLUS else "single_model"


def _train_bootstrap_model(
    detector: AnomalyDetector,
    x: np.ndarray,
    bootstrap_indices: np.ndarray,
    seed: int | None,
) -> AnomalyDetector:
    """Train a single bootstrap model (module-level for safe pickling).

    This function is defined at module level to ensure clean pickling
    when used with ProcessPoolExecutor, avoiding capture of unnecessary
    class state.

    Args:
        detector: Base detector to clone and train.
        x: Full training data array.
        bootstrap_indices: Indices for bootstrap sample.
        seed: Random seed for reproducibility.

    Returns:
        Trained detector model.
    """
    model = deepcopy(detector)
    if hasattr(model, "set_params"):
        try:
            model.set_params(random_state=seed)
        except (TypeError, ValueError):
            pass  # Detector may not support random_state parameter
    model.fit(x[bootstrap_indices])
    return model


class JackknifeBootstrap(BaseStrategy):
    """Bootstrap and out-of-bag calibration for conformal anomaly scoring.

    Each bootstrap replica is fitted on a sample drawn with replacement. Every
    reference observation receives a calibration score aggregated over replicas
    for which that observation was out of bag. In ``"plus"`` mode, test scores
    are aggregated over all retained replicas.

    The construction is inspired by jackknife+-after-bootstrap (JaB+), but this
    class produces anomaly scores and p-values rather than the predictive
    intervals studied by the JaB+ theorem. Do not infer an interval-coverage
    guarantee solely from the class name.

    Args:
        n_bootstraps: Number of bootstrap iterations. Defaults to 100.
        aggregation_method: How to aggregate OOB predictions ("mean" or "median").
            Defaults to "mean".
        mode: Model retention mode (`"plus"` or `"single_model"`). Equivalent
            ``ConformalMode`` values are accepted. Defaults to `"plus"`.

    References:
        Kim, Byol, Chen Xu, and Rina Foygel Barber. "Predictive Inference Is Free
        with the Jackknife+-after-Bootstrap." NeurIPS 2020.
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        aggregation_method: BootstrapAggregationMethod = "mean",
        mode: ConformalModeInput = "plus",
    ) -> None:
        super().__init__(mode=mode)

        if n_bootstraps < 2:
            exc = ValueError(
                f"Number of bootstraps must be at least 2, got {n_bootstraps}."
            )
            exc.add_note(f"Received n_bootstraps={n_bootstraps}, which is invalid.")
            raise exc

        normalized_aggregation_method = normalize_bootstrap_aggregation_method(
            aggregation_method
        )

        if self._mode is ConformalMode.SINGLE_MODEL:
            _bootstrap_logger.warning(
                "Setting mode=ConformalMode.SINGLE_MODEL may compromise conformal "
                "validity. mode=ConformalMode.PLUS is recommended."
            )

        self._n_bootstraps: int = n_bootstraps
        self._aggregation_method: BootstrapAggregationMethod = (
            normalized_aggregation_method
        )

        self._detector_list: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_ids: list[int] = []

        # Internal state
        self._bootstrap_models: list[AnomalyDetector | None] = []
        self._oob_mask: np.ndarray = np.array([])

    @ensure_numpy_array
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
        n_jobs: int | None = None,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fit bootstrap replicas and compute out-of-bag calibration scores.

        Args:
            x: Input data matrix.
            detector: The base anomaly detector.
            seed: Random seed for reproducibility. Defaults to None.
            weighted: Accepted for the shared strategy interface. Calibration
                indices already include every input row in this strategy.
            n_jobs: Number of parallel jobs. Use -1 for all available cores.
                Defaults to None (sequential).

        Returns:
            Tuple of (list of trained detectors, calibration scores array).
        """
        n_samples = len(x)
        generator = np.random.default_rng(seed)

        _bootstrap_logger.info(
            f"Bootstrap (JaB+): {n_samples:,} samples, "
            f"{self._n_bootstraps:,} iterations"
        )

        self._bootstrap_models = [None] * self._n_bootstraps
        all_bootstrap_indices, self._oob_mask = self._generate_bootstrap_indices(
            generator, n_samples
        )

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        elif n_jobs is not None and n_jobs < 1:
            raise ValueError(
                f"n_jobs must be None, -1, or a positive integer; got {n_jobs}."
            )

        if n_jobs is None or n_jobs == 1:
            bootstrap_iterator = (
                tqdm(range(self._n_bootstraps), desc="Calibration")
                if _bootstrap_logger.isEnabledFor(logging.INFO)
                else range(self._n_bootstraps)
            )
            for i in bootstrap_iterator:
                bootstrap_indices = all_bootstrap_indices[i]
                model = _train_bootstrap_model(detector, x, bootstrap_indices, seed)
                self._bootstrap_models[i] = model
        else:
            self._train_models_parallel(
                detector, x, all_bootstrap_indices, seed, n_jobs
            )

        oob_scores = self._compute_oob_scores(x)

        self._calibration_set = oob_scores
        self._calibration_ids = list(range(n_samples))

        if self._mode is ConformalMode.PLUS:
            self._detector_list = self._bootstrap_models.copy()
        else:
            final_model = deepcopy(detector)
            if hasattr(final_model, "set_params"):
                try:
                    final_model.set_params(random_state=seed)
                except (TypeError, ValueError):
                    pass  # Detector may not support random_state parameter
            final_model.fit(x)
            self._detector_list = [final_model]

        return self._detector_list, self._calibration_set

    def _generate_bootstrap_indices(
        self, generator: np.random.Generator, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate bootstrap indices with guaranteed OOB coverage."""
        if n_samples < 2:
            raise ValueError("JackknifeBootstrap requires at least 2 samples.")

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

    def _train_models_parallel(
        self,
        detector: AnomalyDetector,
        x: pd.DataFrame | np.ndarray,
        all_bootstrap_indices: np.ndarray,
        seed: int | None,
        n_jobs: int,
    ) -> None:
        """Train bootstrap models in parallel.

        Uses module-level _train_bootstrap_model function to ensure clean
        pickling without capturing unnecessary class state.
        """
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _train_bootstrap_model,
                    detector,
                    x,
                    all_bootstrap_indices[i],
                    seed,
                ): i
                for i in range(self._n_bootstraps)
            }

            future_iterator = (
                tqdm(
                    as_completed(futures), total=self._n_bootstraps, desc="Calibration"
                )
                if _bootstrap_logger.isEnabledFor(logging.INFO)
                else as_completed(futures)
            )
            for future in future_iterator:
                i = futures[future]
                self._bootstrap_models[i] = future.result()

    def _aggregate_predictions(self, predictions: list | np.ndarray) -> float:
        """Aggregate predictions using configured method."""
        if len(predictions) == 0:
            return np.nan

        match self._aggregation_method:
            case "mean":
                return np.mean(predictions)
            case "median":
                return np.median(predictions)
            case _:
                raise ValueError(f"Unsupported aggregation: {self._aggregation_method}")

    def _compute_oob_scores(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute out-of-bag calibration scores."""
        n_samples = len(x)
        all_predictions = [[] for _ in range(n_samples)]

        for model_idx, model in enumerate(self._bootstrap_models):
            oob_samples = self._oob_mask[model_idx]
            oob_indices = np.where(oob_samples)[0]

            if len(oob_indices) > 0:
                oob_predictions = model.decision_function(x[oob_indices])
                for idx, pred in zip(oob_indices, oob_predictions):
                    all_predictions[idx].append(pred)

        # Check coverage
        no_predictions = np.array([len(preds) == 0 for preds in all_predictions])
        if np.any(no_predictions):
            raise ValueError(
                f"Samples {np.where(no_predictions)[0]} have no OOB predictions. "
                "Consider increasing n_bootstraps."
            )

        oob_scores = np.array(
            [self._aggregate_predictions(preds) for preds in all_predictions]
        )
        return oob_scores

    @property
    def calibration_ids(self) -> list[int]:
        """Indices used for calibration (all samples in JaB+)."""
        return self._calibration_ids.copy()

    @property
    def n_bootstraps(self) -> int:
        """Number of bootstrap iterations."""
        return self._n_bootstraps

    @property
    def aggregation_method(self) -> BootstrapAggregationMethod:
        """Aggregation method for OOB predictions."""
        return self._aggregation_method


__all__ = [
    "BaseStrategy",
    "CrossValidation",
    "DerandomizedSplits",
    "JackknifeBootstrap",
    "Split",
]
