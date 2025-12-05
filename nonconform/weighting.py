"""Weight estimation for covariate shift correction in weighted conformal prediction.

This module provides weight estimators that compute importance weights to correct
for covariate shift between calibration and test distributions. They estimate
density ratios w(x) = p_test(x) / p_calib(x) which are used to reweight conformal
scores for better coverage guarantees under distribution shift.

Classes:
    BaseWeightEstimator: Abstract base class for weight estimators.
    IdentityWeightEstimator: Returns uniform weights (no covariate shift).
    SklearnWeightEstimator: Universal wrapper for sklearn probabilistic classifiers.
    BootstrapBaggedWeightEstimator: Bootstrap-bagged wrapper for robust estimation.

Factory functions:
    logistic_weight_estimator: Create estimator using Logistic Regression.
    forest_weight_estimator: Create estimator using Random Forest.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Protocol

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

DEFAULT_CLIP_BOUNDS = (0.35, 45.0)
EPSILON = 1e-9
_bagged_logger = logging.getLogger("nonconform.weighting.bagged")


class ProbabilisticClassifier(Protocol):
    """Protocol for classifiers that support probability estimation.

    This protocol defines the interface for sklearn-compatible classifiers
    that can produce probability estimates for weight computation.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> ProbabilisticClassifier:
        """Fit the classifier on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).

        Returns:
            The fitted classifier instance.
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability estimates of shape (n_samples, n_classes).
        """
        ...

    classes_: np.ndarray


class BaseWeightEstimator(ABC):
    """Abstract base class for weight estimators in weighted conformal prediction.

    Weight estimators compute importance weights to correct for covariate shift
    between calibration and test distributions. They estimate density ratios
    w(x) = p_test(x) / p_calib(x) which are used to reweight conformal scores
    for better coverage guarantees under distribution shift.

    Subclasses must implement fit(), _get_stored_weights(), and _score_new_data()
    to provide specific weight estimation strategies.
    """

    @abstractmethod
    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Estimate density ratio weights."""
        pass

    def get_weights(
        self,
        calibration_samples: np.ndarray | None = None,
        test_samples: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return density ratio weights for calibration and test data.

        Args:
            calibration_samples: Optional calibration data to score. If provided,
                computes weights for this data using the fitted model. If None,
                returns stored weights from fit(). Must provide both or neither.
            test_samples: Optional test data to score. If provided, computes
                weights for this data using the fitted model. If None, returns
                stored weights from fit(). Must provide both or neither.

        Returns:
            Tuple of (calibration_weights, test_weights) as numpy arrays.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If only one of calibration_samples/test_samples is provided.
        """
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("Must call fit() before get_weights()")

        if (calibration_samples is None) != (test_samples is None):
            raise ValueError(
                "Must provide both calibration_samples and test_samples, or neither. "
                "Cannot score only one set."
            )

        if calibration_samples is None:
            return self._get_stored_weights()
        else:
            return self._score_new_data(calibration_samples, test_samples)

    @abstractmethod
    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored weights from fit()."""
        pass

    @abstractmethod
    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score new data using the fitted model."""
        pass

    def set_seed(self, seed: int | None) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value or None.
        """
        self._seed = seed

    # --------------------------------------------------------------------------
    # Helper methods for subclasses (shared logic)
    # --------------------------------------------------------------------------

    @staticmethod
    def _prepare_training_data(
        calibration_samples: np.ndarray,
        test_samples: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare labeled and shuffled training data for classifier-based estimation.

        Labels calibration samples as 0 and test samples as 1, then shuffles.

        Args:
            calibration_samples: Calibration data samples.
            test_samples: Test data samples.
            seed: Random seed for shuffling.

        Returns:
            Tuple of (X, y) arrays ready for classifier training.
        """
        # Label calibration samples as 0, test samples as 1
        calib_labeled = np.hstack(
            (calibration_samples, np.zeros((calibration_samples.shape[0], 1)))
        )
        test_labeled = np.hstack((test_samples, np.ones((test_samples.shape[0], 1))))

        joint_labeled = np.vstack((calib_labeled, test_labeled))
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(joint_labeled)

        x_joint = joint_labeled[:, :-1]
        y_joint = joint_labeled[:, -1]

        return x_joint, y_joint

    @staticmethod
    def _compute_clip_bounds(
        w_calib: np.ndarray,
        w_test: np.ndarray,
        clip_quantile: float | None,
    ) -> tuple[float, float]:
        """Compute clipping bounds for weight stabilization.

        Args:
            w_calib: Calibration weights.
            w_test: Test weights.
            clip_quantile: Quantile for adaptive clipping (e.g., 0.05 clips to
                5th-95th percentile). If None, uses default fixed bounds.

        Returns:
            Tuple of (lower_bound, upper_bound) for clipping.
        """
        if clip_quantile is not None:
            all_weights = np.concatenate([w_calib, w_test])
            lower_bound = np.percentile(all_weights, clip_quantile * 100)
            upper_bound = np.percentile(all_weights, (1 - clip_quantile) * 100)
            return (lower_bound, upper_bound)
        return DEFAULT_CLIP_BOUNDS

    @staticmethod
    def _clip_weights(
        w_calib: np.ndarray,
        w_test: np.ndarray,
        clip_bounds: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply clipping to stabilize weights.

        Args:
            w_calib: Calibration weights.
            w_test: Test weights.
            clip_bounds: Tuple of (lower, upper) bounds.

        Returns:
            Tuple of clipped (w_calib, w_test) arrays.
        """
        return np.clip(w_calib, *clip_bounds), np.clip(w_test, *clip_bounds)


class IdentityWeightEstimator(BaseWeightEstimator):
    """Identity weight estimator that returns uniform weights.

    This estimator assumes no covariate shift and returns weights of 1.0
    for all samples. Useful as a baseline or when covariate shift is known
    to be minimal.

    This effectively makes weighted conformal prediction equivalent to
    standard conformal prediction.
    """

    def __init__(self) -> None:
        self._n_calib = 0
        self._n_test = 0
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the identity weight estimator.

        Args:
            calibration_samples: Array of calibration data samples.
            test_samples: Array of test data samples.
        """
        self._n_calib = calibration_samples.shape[0]
        self._n_test = test_samples.shape[0]
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return uniform weights of 1.0 for stored sizes."""
        calib_weights = np.ones(self._n_calib, dtype=np.float64)
        test_weights = np.ones(self._n_test, dtype=np.float64)
        return calib_weights, test_weights

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return uniform weights of 1.0 for provided data."""
        calib_weights = np.ones(calibration_samples.shape[0], dtype=np.float64)
        test_weights = np.ones(test_samples.shape[0], dtype=np.float64)
        return calib_weights, test_weights


class SklearnWeightEstimator(BaseWeightEstimator):
    """Universal wrapper for any sklearn-compatible probabilistic classifier.

    Adheres to the standard sklearn 'Meta-Estimator' pattern.
    Accepts a configured estimator instance and clones it for cross-validation safety.

    Args:
        base_estimator: Configured sklearn classifier instance with predict_proba
            support. Defaults to LogisticRegression.
        clip_quantile: Quantile for weight clipping (e.g., 0.05 clips to 5th-95th
            percentile). Defaults to 0.05.

    Raises:
        ValueError: If base_estimator does not implement predict_proba.

    Examples:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Default (LogisticRegression)
        estimator = SklearnWeightEstimator()

        # Custom with pipeline
        estimator = SklearnWeightEstimator(
            base_estimator=make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced")
            )
        )

        # Random Forest
        estimator = SklearnWeightEstimator(
            base_estimator=RandomForestClassifier(n_estimators=100, max_depth=5)
        )
        ```
    """

    # Label convention: Calibration = 0, Test = 1
    CALIB_LABEL = 0
    TEST_LABEL = 1

    def __init__(
        self,
        base_estimator: ProbabilisticClassifier | BaseEstimator | None = None,
        clip_quantile: float = 0.05,
    ) -> None:
        # Default to a sane baseline if nothing is provided
        # Use explicit None check to avoid truthiness evaluation of sklearn estimators
        # (unfitted ensemble estimators raise AttributeError on __len__)
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else LogisticRegression(solver="liblinear")
        )
        self.clip_quantile = clip_quantile

        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError(
                f"The provided base_estimator {type(self.base_estimator).__name__} "
                "does not implement 'predict_proba'. Density estimation requires "
                "probability scores. Use SVC(probability=True) or similar."
            )

        # Seed inheritance attribute (may be set by ConformalDetector)
        self._seed: int | None = None

        self.estimator_: ProbabilisticClassifier | None = None
        self._test_class_idx: int | None = None  # Column index for P(Test)
        self._w_calib: np.ndarray | None = None
        self._w_test: np.ndarray | None = None
        self._clip_bounds: tuple[float, float] | None = None
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the weight estimator on calibration and test samples.

        Args:
            calibration_samples: Array of calibration data samples.
            test_samples: Array of test data samples.

        Raises:
            ValueError: If calibration_samples is empty.
        """
        if calibration_samples.shape[0] == 0:
            raise ValueError("Calibration samples are empty. Cannot compute weights.")

        # Prepare data (Calib=0, Test=1 labels)
        x_joint, y_joint = self._prepare_training_data(
            calibration_samples, test_samples, self._seed
        )

        self.estimator_ = clone(self.base_estimator)
        if self._seed is not None:
            self._apply_seed_to_estimator(self.estimator_, self._seed)
        self.estimator_.fit(x_joint, y_joint)

        # sklearn sorts classes_ - get correct column index for P(Test)
        self._test_class_idx = int(
            np.where(self.estimator_.classes_ == self.TEST_LABEL)[0][0]
        )

        w_calib, w_test = self._compute_weights(calibration_samples, test_samples)
        self._clip_bounds = self._compute_clip_bounds(
            w_calib, w_test, self.clip_quantile
        )
        self._w_calib, self._w_test = self._clip_weights(
            w_calib, w_test, self._clip_bounds
        )
        self._is_fitted = True

    def _compute_weights(
        self, calib_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute density ratio weights using verified class index."""
        calib_prob = self.estimator_.predict_proba(calib_samples)
        test_prob = self.estimator_.predict_proba(test_samples)

        # Use verified column indices (not hardcoded 0/1)
        calib_idx = 1 - self._test_class_idx  # Opposite of test
        test_idx = self._test_class_idx

        # w(z) = P(Test|z) / P(Calib|z)
        w_calib = calib_prob[:, test_idx] / (calib_prob[:, calib_idx] + EPSILON)
        w_test = test_prob[:, test_idx] / (test_prob[:, calib_idx] + EPSILON)
        return w_calib, w_test

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored weights from fit()."""
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score new data using fitted model."""
        w_calib, w_test = self._compute_weights(calibration_samples, test_samples)
        return self._clip_weights(w_calib, w_test, self._clip_bounds)

    @staticmethod
    def _apply_seed_to_estimator(estimator: ProbabilisticClassifier, seed: int) -> None:
        """Apply random seed to sklearn estimator or pipeline.

        Args:
            estimator: Sklearn estimator or pipeline to configure.
            seed: Random seed value to set.
        """
        from sklearn.pipeline import Pipeline

        if isinstance(estimator, Pipeline):
            for _, step in estimator.steps:
                if hasattr(step, "random_state"):
                    step.random_state = seed
        elif hasattr(estimator, "random_state"):
            estimator.random_state = seed


class BootstrapBaggedWeightEstimator(BaseWeightEstimator):
    """Bootstrap-bagged wrapper for weight estimators with instance-wise aggregation.

    This estimator wraps any base weight estimator and applies bootstrap bagging
    to create more stable, robust weight estimates. It's particularly useful when
    the calibration set is much larger than the test batch (or vice versa).

    The algorithm:
    1. For each bootstrap iteration:
       - Resample BOTH sets to balanced sample size (min of calibration and test sizes)
       - Fit the base estimator on the balanced bootstrap sample
       - Score ALL original instances using the fitted model (perfect coverage)
       - Store log(weights) for each instance
    2. After all iterations:
       - Aggregate instance-wise weights using geometric mean (average in log-space)
       - Apply clipping to maintain boundedness for theoretical guarantees

    Seed inheritance:
        This class uses the `_seed` attribute pattern for automatic seed
        inheritance from ConformalDetector.

    Args:
        base_estimator: Any BaseWeightEstimator instance.
        n_bootstrap: Number of bootstrap iterations. Defaults to 100.
        clip_bounds: Fixed clipping bounds. Defaults to (0.35, 45.0).
        clip_quantile: Quantile for adaptive clipping. Defaults to 0.05.

    References:
        Jin, Ying, and Emmanuel J. Candès. "Selection by Prediction with Conformal
        p-values." Journal of Machine Learning Research 24.244 (2023): 1-41.
    """

    def __init__(
        self,
        base_estimator: BaseWeightEstimator,
        n_bootstrap: int = 100,
        clip_bounds: tuple[float, float] = (0.35, 45.0),
        clip_quantile: float = 0.05,
    ) -> None:
        if n_bootstrap < 1:
            raise ValueError(
                f"n_bootstrap must be at least 1, got {n_bootstrap}. "
                f"Typical values are 50-200 for stable weight estimation."
            )
        if clip_quantile is not None and not (0 < clip_quantile < 0.5):
            raise ValueError(
                f"clip_quantile must be in (0, 0.5), got {clip_quantile}. "
                f"Common values are 0.05 (5th-95th percentiles) or 0.01."
            )

        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.clip_bounds = clip_bounds
        self.clip_quantile = clip_quantile

        # Seed inheritance attribute (set by ConformalDetector)
        self._seed: int | None = None

        self._w_calib: np.ndarray | None = None
        self._w_test: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the bagged weight estimator with perfect instance coverage.

        Args:
            calibration_samples: Array of calibration data samples.
            test_samples: Array of test data samples.

        Raises:
            ValueError: If calibration_samples is empty.
        """
        if calibration_samples.shape[0] == 0:
            raise ValueError("Calibration samples are empty. Cannot compute weights.")

        n_calib, n_test = len(calibration_samples), len(test_samples)
        sample_size = min(n_calib, n_test)
        rng = np.random.default_rng(self._seed)

        if _bagged_logger.isEnabledFor(logging.INFO):
            _bagged_logger.info(
                f"Bootstrap: n_calib={n_calib}, n_test={n_test}, "
                f"sample_size={sample_size}, n_bootstrap={self.n_bootstrap}. "
                f"Perfect coverage: all instances weighted in all iterations."
            )

        # Online accumulation: sum log-weights (memory efficient)
        sum_log_weights_calib = np.zeros(n_calib)
        sum_log_weights_test = np.zeros(n_test)

        bootstrap_iterator = (
            tqdm(range(self.n_bootstrap), desc="Weighting")
            if _bagged_logger.isEnabledFor(logging.INFO)
            else range(self.n_bootstrap)
        )

        for i in bootstrap_iterator:
            # Resample both sets for balanced comparison
            calib_indices = rng.choice(n_calib, size=sample_size, replace=True)
            test_indices = rng.choice(n_test, size=sample_size, replace=True)
            x_calib_boot = calibration_samples[calib_indices]
            x_test_boot = test_samples[test_indices]

            # Create base estimator with iteration-specific seed
            base_est = deepcopy(self.base_estimator)
            if self._seed is not None:
                if hasattr(base_est, "seed"):
                    base_est.seed = hash((i, self._seed)) % (2**32)
                if hasattr(base_est, "_seed"):
                    base_est._seed = hash((i, self._seed)) % (2**32)

            # Fit on bootstrap sample, then score ALL original instances
            base_est.fit(x_calib_boot, x_test_boot)
            w_c_all, w_t_all = base_est.get_weights(calibration_samples, test_samples)

            # Accumulate log-weights for geometric mean aggregation
            sum_log_weights_calib += np.log(w_c_all)
            sum_log_weights_test += np.log(w_t_all)

        # Geometric mean aggregation: exp(mean(log-weights))
        w_calib_final = np.exp(sum_log_weights_calib / self.n_bootstrap)
        w_test_final = np.exp(sum_log_weights_test / self.n_bootstrap)

        # Apply clipping after aggregation (use base class static method)
        clip_min, clip_max = BaseWeightEstimator._compute_clip_bounds(
            w_calib_final, w_test_final, self.clip_quantile
        )
        if self.clip_quantile is None:
            clip_min, clip_max = self.clip_bounds

        self._w_calib = np.clip(w_calib_final, clip_min, clip_max)
        self._w_test = np.clip(w_test_final, clip_min, clip_max)
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return instance-wise aggregated weights from fit()."""
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bagged weights are tied to samples seen during fit().

        Raises:
            NotImplementedError: Bagged estimator cannot rescore new data.
        """
        if (
            calibration_samples.shape[0] != self._w_calib.shape[0]
            or test_samples.shape[0] != self._w_test.shape[0]
        ):
            raise NotImplementedError(
                "BootstrapBaggedWeightEstimator cannot rescore new data. "
                "Refit the estimator with the desired calibration/test samples."
            )
        return self._get_stored_weights()

    @property
    def weight_counts(self) -> str:
        """Return diagnostic info about instance-wise weight coverage."""
        if not self._is_fitted:
            return "Not fitted yet"
        return (
            f"Bootstrap iterations: {self.n_bootstrap}\n"
            f"Calibration instances: {len(self._w_calib)}\n"
            f"Test instances: {len(self._w_test)}"
        )


# -----------------------------------------------------------------------------
# Factory functions for behavioral equivalence with old API
# -----------------------------------------------------------------------------


def logistic_weight_estimator(
    regularization: str | float = "auto",
    clip_quantile: float = 0.05,
    class_weight: str | dict = "balanced",
    max_iter: int = 1000,
) -> SklearnWeightEstimator:
    """Create weight estimator using Logistic Regression.

    This factory function provides behavioral equivalence with the old
    LogisticWeightEstimator class.

    Note:
        When used with ConformalDetector, the detector's seed is automatically
        propagated to the weight estimator for reproducibility.

    Args:
        regularization: Regularization parameter. If 'auto', uses C=1.0.
            If float, uses as C parameter.
        clip_quantile: Quantile for weight clipping. Defaults to 0.05.
        class_weight: Class weights for LogisticRegression. Defaults to 'balanced'.
        max_iter: Maximum iterations for solver convergence. Defaults to 1000.

    Returns:
        Configured SklearnWeightEstimator instance.

    Examples:
        ```python
        estimator = logistic_weight_estimator(regularization=0.5)
        estimator.fit(calib_samples, test_samples)
        w_calib, w_test = estimator.get_weights()
        ```
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    c_param = 1.0 if regularization == "auto" else float(regularization)
    base_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            class_weight=class_weight,
        ),
    )
    return SklearnWeightEstimator(
        base_estimator=base_estimator, clip_quantile=clip_quantile
    )


def forest_weight_estimator(
    n_estimators: int = 100,
    max_depth: int | None = 5,
    min_samples_leaf: int = 10,
    clip_quantile: float = 0.05,
) -> SklearnWeightEstimator:
    """Create weight estimator using Random Forest.

    This factory function provides behavioral equivalence with the old
    ForestWeightEstimator class.

    Note:
        When used with ConformalDetector, the detector's seed is automatically
        propagated to the weight estimator for reproducibility.

    Args:
        n_estimators: Number of trees in the forest. Defaults to 100.
        max_depth: Maximum depth of trees. Defaults to 5.
        min_samples_leaf: Minimum samples at leaf node. Defaults to 10.
        clip_quantile: Quantile for weight clipping. Defaults to 0.05.

    Returns:
        Configured SklearnWeightEstimator instance.

    Examples:
        ```python
        estimator = forest_weight_estimator(n_estimators=200)
        estimator.fit(calib_samples, test_samples)
        w_calib, w_test = estimator.get_weights()
        ```
    """
    from sklearn.ensemble import RandomForestClassifier

    base_estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
    )
    return SklearnWeightEstimator(
        base_estimator=base_estimator, clip_quantile=clip_quantile
    )


__all__ = [
    "DEFAULT_CLIP_BOUNDS",
    "EPSILON",
    "BaseWeightEstimator",
    "BootstrapBaggedWeightEstimator",
    "IdentityWeightEstimator",
    "ProbabilisticClassifier",
    "SklearnWeightEstimator",
    "forest_weight_estimator",
    "logistic_weight_estimator",
]
