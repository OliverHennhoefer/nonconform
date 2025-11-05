"""Bootstrap-bagged weight estimator for robust covariate shift correction.

This module implements a wrapper that applies bootstrap bagging to any weight
estimator, providing more stable and robust weight estimates when calibration
and test set sizes are imbalanced.
"""

import logging
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from nonconform.detection.weight.base import BaseWeightEstimator
from nonconform.utils.func.logger import get_logger


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

    This approach ensures every instance receives exactly n_bootstrap weight estimates,
    providing deterministic coverage and eliminating asymmetry between calibration
    and test sets.

    Seed inheritance:
        This class uses the `_seed` attribute pattern for automatic seed
        inheritance from ConformalDetector. Users should NOT pass a seed
        parameter - it will be set automatically by the parent detector.

    Args:
        base_estimator: Any BaseWeightEstimator instance (e.g., LogisticWeightEstimator,
            ForestWeightEstimator). This estimator will be used for each bootstrap
            iteration.
        n_bootstrap: Number of bootstrap iterations. Higher values provide more
            stable weights but increase computational cost. Defaults to 100.
        clip_bounds: Fixed clipping bounds applied after aggregation as
            (min_weight, max_weight). Defaults to (0.35, 45.0).
        clip_quantile: Quantile for adaptive weight clipping. If provided,
            clips to (quantile, 1-quantile) percentiles instead of fixed bounds.
            Defaults to 0.05 (clips to 5th-95th percentiles).

    Attributes:
        _seed: Random seed inherited from ConformalDetector (set automatically).
        _w_calib: Final aggregated weights for calibration instances.
        _w_test: Final aggregated weights for test instances.
        _is_fitted: Whether fit() has been called.

    Examples:
        ```python
        from pyod.models.iforest import IForest
        from nonconform.detection import ConformalDetector
        from nonconform.detection.weight import (
            LogisticWeightEstimator,
            BootstrapBaggedWeightEstimator,
        )
        from nonconform.strategy import Split

        # Create bagged weight estimator (seed inherited from detector)
        detector = ConformalDetector(
            detector=IForest(),
            strategy=Split(n_calib=1000),
            weight_estimator=BootstrapBaggedWeightEstimator(
                base_estimator=LogisticWeightEstimator(),
                n_bootstrap=100,
            ),
            seed=42,  # Automatically propagated to weight estimator
        )

        detector.fit(X_train)
        p_values = detector.predict(X_test)
        ```

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
    ):
        """Initialize the bootstrap-bagged weight estimator.

        Args:
            base_estimator: Base weight estimator to wrap.
            n_bootstrap: Number of bootstrap iterations (default: 100).
            clip_bounds: Fixed clipping bounds (default: (0.35, 45.0)).
            clip_quantile: Adaptive quantile clipping (default: 0.05).

        Raises:
            ValueError: If n_bootstrap < 1.
            ValueError: If clip_quantile not in (0, 0.5).
        """
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

        # Weight storage
        self._w_calib: np.ndarray | None = None
        self._w_test: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the bagged weight estimator with perfect instance coverage.

        For each bootstrap iteration:
        - Resamples BOTH sets to balanced sample size
        - Fits the base estimator on the balanced bootstrap sample
        - Scores ALL original instances using the fitted model
        - Stores log(weights) for each instance

        After all iterations:
        - Aggregates using geometric mean (exp of mean log-weights)
        - Applies clipping to maintain bounded weights

        Every instance receives exactly n_bootstrap weight estimates,
        ensuring symmetric coverage regardless of set size ratios.

        Args:
            calibration_samples: Array of calibration data samples of shape
                (n_calib, n_features).
            test_samples: Array of test data samples of shape (n_test, n_features).

        Raises:
            ValueError: If calibration_samples is empty.
        """
        if calibration_samples.shape[0] == 0:
            raise ValueError("Calibration samples are empty. Cannot compute weights.")

        n_calib, n_test = len(calibration_samples), len(test_samples)
        sample_size = min(n_calib, n_test)  # Always use balanced approach
        rng = np.random.default_rng(self._seed)
        logger = get_logger("detection.weight.bagged")

        # Log coverage info concisely
        if logger.isEnabledFor(logging.INFO):
            # Perfect coverage: every instance weighted in every iteration
            logger.info(
                f"Bootstrap: n_calib={n_calib}, n_test={n_test}, "
                f"sample_size={sample_size}, n_bootstrap={self.n_bootstrap}. "
                f"Perfect coverage: all instances weighted in all iterations."
            )

        # Online accumulation: sum log-weights (memory efficient)
        sum_log_weights_calib = np.zeros(n_calib)
        sum_log_weights_test = np.zeros(n_test)

        # Intelligent tqdm: show progress bar if logging level is INFO or higher
        bootstrap_iterator = (
            tqdm(range(self.n_bootstrap), desc="Weighting")
            if logger.isEnabledFor(logging.INFO)
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
            if self._seed is not None and hasattr(base_est, "seed"):
                base_est.seed = hash((i, self._seed)) % (2**32)

            # Fit on bootstrap sample, then score ALL original instances
            base_est.fit(x_calib_boot, x_test_boot)
            w_c_all, w_t_all = base_est.get_weights(calibration_samples, test_samples)

            # Accumulate log-weights for geometric mean aggregation
            sum_log_weights_calib += np.log(w_c_all)
            sum_log_weights_test += np.log(w_t_all)

        # Geometric mean aggregation: exp(mean(log-weights))
        # Every instance has exactly n_bootstrap weights (perfect coverage)
        w_calib_final = np.exp(sum_log_weights_calib / self.n_bootstrap)
        w_test_final = np.exp(sum_log_weights_test / self.n_bootstrap)

        # Apply clipping after aggregation
        clip_min, clip_max = self._compute_clip_bounds(w_calib_final, w_test_final)
        self._w_calib = np.clip(w_calib_final, clip_min, clip_max)
        self._w_test = np.clip(w_test_final, clip_min, clip_max)
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return instance-wise aggregated weights from fit()."""
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return stored weights when callers pass the original data.

        Bagged weights are tied to the calibration/test samples seen during fit().
        We only allow scoring requests that match the fitted sample sizes (the
        default path in BaseWeightEstimator); any other inputs require a refit.
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

    def _compute_clip_bounds(
        self, w_calib: np.ndarray, w_test: np.ndarray
    ) -> tuple[float, float]:
        """Compute clipping bounds for weights.

        Uses either adaptive quantile-based clipping or fixed bounds.

        Args:
            w_calib: Calibration weights before clipping.
            w_test: Test weights before clipping.

        Returns:
            Tuple of (lower_bound, upper_bound) for weight clipping.
        """
        if self.clip_quantile is not None:
            # Adaptive clipping based on percentiles
            all_weights = np.concatenate([w_calib, w_test])
            lower_bound = np.percentile(all_weights, self.clip_quantile * 100)
            upper_bound = np.percentile(all_weights, (1 - self.clip_quantile) * 100)
            return lower_bound, upper_bound
        else:
            # Fixed clipping bounds
            return self.clip_bounds

    @property
    def weight_counts(self) -> str:
        """Return diagnostic info about instance-wise weight coverage.

        This is useful for verifying that instances are getting sufficient
        coverage across bootstrap iterations.

        Returns:
            String describing weight statistics.
        """
        if not self._is_fitted:
            return "Not fitted yet"

        # In the current implementation, we don't store counts separately,
        # but we can infer that with balanced sampling and replacement,
        # each instance in the larger set appears ~n_bootstrap * (sample_size/n_larger)
        # times on average
        return (
            f"Bootstrap iterations: {self.n_bootstrap}\n"
            f"Calibration instances: {len(self._w_calib)}\n"
            f"Test instances: {len(self._w_test)}"
        )
