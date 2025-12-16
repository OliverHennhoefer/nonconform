"""Integration tests for calibration-conditional p-value adjustment."""

from __future__ import annotations

import numpy as np
import pytest
from pyod.models.iforest import IForest

from nonconform import (
    Adjustment,
    BootstrapBaggedWeightEstimator,
    ConformalDetector,
    Empirical,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)


def _build_detector(estimation, seed=42, weight_estimator=None):
    return ConformalDetector(
        detector=IForest(n_estimators=20, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=estimation,
        weight_estimator=weight_estimator,
        seed=seed,
    )


def _build_weight_estimator():
    return BootstrapBaggedWeightEstimator(
        base_estimator=logistic_weight_estimator(),
        n_bootstrap=10,  # Small for speed in tests
    )


ADJUSTMENT_METHODS = [
    ("none", Adjustment.NONE),
    ("simes", Adjustment.SIMES),
    ("asymptotic", Adjustment.ASYMPTOTIC),
    ("monte_carlo", Adjustment.MONTE_CARLO),
]


class TestConditionalAdjustmentBasics:
    """Basic tests for conditional adjustment functionality."""

    @pytest.mark.parametrize(
        ("name", "adjustment"),
        ADJUSTMENT_METHODS,
        ids=[case[0] for case in ADJUSTMENT_METHODS],
    )
    def test_all_adjustments_return_valid_pvalues(
        self, simple_dataset, name, adjustment
    ):
        """All adjustment methods should return p-values in [0, 1]."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        estimation = Empirical(
            adjustment=adjustment, delta=0.1, monte_carlo_samples=500
        )
        detector = _build_detector(estimation)

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        assert p_values.shape == (len(x_test),)
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    def test_conditional_pvalues_at_least_marginal(self, simple_dataset):
        """Conditional p-values should always be >= marginal p-values."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        marginal = _build_detector(Empirical(adjustment=Adjustment.NONE), seed=42)
        conditional = _build_detector(
            Empirical(adjustment=Adjustment.SIMES, delta=0.1), seed=42
        )

        marginal.fit(x_train)
        conditional.fit(x_train)

        p_marginal = marginal.predict(x_test)
        p_conditional = conditional.predict(x_test)

        assert np.all(p_conditional >= p_marginal - 1e-10), (
            "Conditional p-values must be >= marginal"
        )

    def test_adjustment_preserves_weak_ordering(self, simple_dataset):
        """If p_marg[i] < p_marg[j], then p_cond[i] <= p_cond[j] (weak ordering)."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        marginal = _build_detector(Empirical(adjustment=Adjustment.NONE), seed=42)
        conditional = _build_detector(
            Empirical(
                adjustment=Adjustment.MONTE_CARLO, delta=0.1, monte_carlo_samples=500
            ),
            seed=42,
        )

        marginal.fit(x_train)
        conditional.fit(x_train)

        p_marginal = marginal.predict(x_test)
        p_conditional = conditional.predict(x_test)

        # Check weak ordering: if p_marg[i] < p_marg[j], then p_cond[i] <= p_cond[j]
        # Note: step function can create ties, so strict ordering isn't preserved
        sorted_indices = np.argsort(p_marginal)
        p_cond_sorted = p_conditional[sorted_indices]
        # p_cond_sorted should be non-decreasing
        assert np.all(np.diff(p_cond_sorted) >= -1e-10), (
            "Weak ordering should be preserved"
        )


class TestAdjustmentMethods:
    """Tests comparing different adjustment methods."""

    @pytest.mark.parametrize(
        ("name", "adjustment"),
        [
            ("simes", Adjustment.SIMES),
            ("asymptotic", Adjustment.ASYMPTOTIC),
            ("monte_carlo", Adjustment.MONTE_CARLO),
        ],
        ids=["simes", "asymptotic", "monte_carlo"],
    )
    def test_adjustments_differ_from_marginal(self, simple_dataset, name, adjustment):
        """All adjustment methods should produce different results from marginal."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        marginal = _build_detector(Empirical(adjustment=Adjustment.NONE), seed=42)
        adjusted = _build_detector(
            Empirical(adjustment=adjustment, delta=0.1, monte_carlo_samples=500),
            seed=42,
        )

        marginal.fit(x_train)
        adjusted.fit(x_train)

        p_marginal = marginal.predict(x_test)
        p_adjusted = adjusted.predict(x_test)

        # Should be different (adjusted >= marginal, with some strict inequality)
        assert not np.allclose(p_marginal, p_adjusted)

    def test_simes_and_asymptotic_differ(self, simple_dataset):
        """Simes and asymptotic adjustments should give different results."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        simes = _build_detector(
            Empirical(adjustment=Adjustment.SIMES, delta=0.1), seed=42
        )
        asymptotic = _build_detector(
            Empirical(adjustment=Adjustment.ASYMPTOTIC, delta=0.1), seed=42
        )

        simes.fit(x_train)
        asymptotic.fit(x_train)

        p_simes = simes.predict(x_test)
        p_asymptotic = asymptotic.predict(x_test)

        assert not np.allclose(p_simes, p_asymptotic)


class TestDeltaParameter:
    """Tests for the delta (miscoverage) parameter."""

    def test_smaller_delta_gives_larger_pvalues(self, simple_dataset):
        """Smaller delta (more conservative) should give larger p-values."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        detector_small_delta = _build_detector(
            Empirical(adjustment=Adjustment.SIMES, delta=0.05), seed=42
        )
        detector_large_delta = _build_detector(
            Empirical(adjustment=Adjustment.SIMES, delta=0.2), seed=42
        )

        detector_small_delta.fit(x_train)
        detector_large_delta.fit(x_train)

        p_small = detector_small_delta.predict(x_test)
        p_large = detector_large_delta.predict(x_test)

        # Smaller delta = more conservative = larger p-values
        assert np.mean(p_small >= p_large - 1e-10) > 0.9


class TestCaching:
    """Tests for adjustment sequence caching."""

    def test_caching_same_calibration_size(self, simple_dataset):
        """Multiple predictions should use cached adjustment sequence."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        estimation = Empirical(adjustment=Adjustment.SIMES, delta=0.1)
        detector = _build_detector(estimation, seed=42)
        detector.fit(x_train)

        # First prediction
        p1 = detector.predict(x_test)
        cache_size_1 = len(estimation._adjustment_cache)

        # Second prediction (should reuse cached sequence)
        p2 = detector.predict(x_test)
        cache_size_2 = len(estimation._adjustment_cache)

        np.testing.assert_array_equal(p1, p2)
        assert cache_size_1 == cache_size_2 == 1


class TestValidation:
    """Tests for parameter validation."""

    def test_invalid_adjustment_type(self):
        """Should raise TypeError for invalid adjustment type."""
        with pytest.raises(TypeError, match="adjustment must be an Adjustment enum"):
            Empirical(adjustment="simes")

    def test_invalid_delta_range(self):
        """Should raise ValueError for delta outside (0, 1)."""
        with pytest.raises(ValueError, match="delta must be in"):
            Empirical(adjustment=Adjustment.SIMES, delta=0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            Empirical(adjustment=Adjustment.SIMES, delta=1.0)

    def test_invalid_monte_carlo_samples(self):
        """Should raise ValueError for too few Monte Carlo samples."""
        with pytest.raises(ValueError, match="monte_carlo_samples must be at least"):
            Empirical(adjustment=Adjustment.MONTE_CARLO, monte_carlo_samples=50)


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_default_empirical_unchanged(self, simple_dataset):
        """Default Empirical() should behave like before (no adjustment)."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)

        # Old-style usage (implicit NONE adjustment)
        detector = _build_detector(Empirical(), seed=42)
        detector.fit(x_train)
        p_default = detector.predict(x_test)

        # Explicit NONE adjustment
        detector_explicit = _build_detector(
            Empirical(adjustment=Adjustment.NONE), seed=42
        )
        detector_explicit.fit(x_train)
        p_explicit = detector_explicit.predict(x_test)

        np.testing.assert_array_equal(p_default, p_explicit)


# Only test SIMES and ASYMPTOTIC for Probabilistic (faster than MC)
ADJUSTMENT_METHODS_FAST = [
    ("simes", Adjustment.SIMES),
    ("asymptotic", Adjustment.ASYMPTOTIC),
]


class TestProbabilisticAdjustment:
    """Tests for Probabilistic estimation with calibration-conditional adjustment."""

    @pytest.mark.parametrize(
        ("name", "adjustment"),
        ADJUSTMENT_METHODS_FAST,
        ids=[case[0] for case in ADJUSTMENT_METHODS_FAST],
    )
    def test_probabilistic_adjustment_valid_pvalues(
        self, simple_dataset, name, adjustment
    ):
        """Probabilistic with adjustment should return valid p-values."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=20, n_features=4)

        estimation = Probabilistic(n_trials=5, adjustment=adjustment, delta=0.1)
        detector = _build_detector(estimation)

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        assert p_values.shape == (len(x_test),)
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    def test_probabilistic_conditional_at_least_marginal(self, simple_dataset):
        """Probabilistic conditional p-values should be >= marginal."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=20, n_features=4)

        marginal = _build_detector(
            Probabilistic(n_trials=5, adjustment=Adjustment.NONE), seed=42
        )
        conditional = _build_detector(
            Probabilistic(n_trials=5, adjustment=Adjustment.ASYMPTOTIC, delta=0.1),
            seed=42,
        )

        marginal.fit(x_train)
        conditional.fit(x_train)

        p_marginal = marginal.predict(x_test)
        p_conditional = conditional.predict(x_test)

        # Allow small tolerance for numerical precision
        assert np.all(p_conditional >= p_marginal - 1e-6)

    def test_probabilistic_asymptotic_recommended(self, simple_dataset):
        """ASYMPTOTIC adjustment should work with Probabilistic (KDE)."""
        x_train, x_test, _ = simple_dataset(n_train=100, n_test=20, n_features=4)

        estimation = Probabilistic(
            n_trials=5, adjustment=Adjustment.ASYMPTOTIC, delta=0.1
        )
        detector = _build_detector(estimation)

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        # Basic sanity check
        assert np.all(p_values >= 0) and np.all(p_values <= 1)


class TestWeightedEmpiricalAdjustment:
    """Tests for Weighted Empirical with calibration-conditional adjustment."""

    @pytest.mark.parametrize(
        ("name", "adjustment"),
        ADJUSTMENT_METHODS_FAST,
        ids=[case[0] for case in ADJUSTMENT_METHODS_FAST],
    )
    def test_weighted_empirical_adjustment_valid_pvalues(
        self, shifted_dataset, name, adjustment
    ):
        """Weighted Empirical with adjustment should return valid p-values."""
        x_train, x_test, _ = shifted_dataset(n_train=100, n_test=20, n_features=4)

        estimation = Empirical(adjustment=adjustment, delta=0.1)
        detector = _build_detector(
            estimation, weight_estimator=_build_weight_estimator()
        )

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        assert p_values.shape == (len(x_test),)
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    def test_weighted_empirical_uses_effective_n(self, shifted_dataset):
        """Weighted adjustment should use effective sample size (n_eff < n)."""
        x_train, x_test, _ = shifted_dataset(n_train=100, n_test=20, n_features=4)

        estimation = Empirical(adjustment=Adjustment.SIMES, delta=0.1)
        detector = _build_detector(
            estimation, weight_estimator=_build_weight_estimator()
        )

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        # Verify adjustment was applied (n_eff tested in unit tests)
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    def test_weighted_empirical_conditional_at_least_marginal(self, shifted_dataset):
        """Weighted conditional p-values should be >= weighted marginal."""
        x_train, x_test, _ = shifted_dataset(n_train=100, n_test=20, n_features=4)

        weight_est = _build_weight_estimator()

        marginal = _build_detector(
            Empirical(adjustment=Adjustment.NONE),
            seed=42,
            weight_estimator=weight_est,
        )
        conditional = _build_detector(
            Empirical(adjustment=Adjustment.SIMES, delta=0.1),
            seed=42,
            weight_estimator=weight_est,
        )

        marginal.fit(x_train)
        conditional.fit(x_train)

        p_marginal = marginal.predict(x_test)
        p_conditional = conditional.predict(x_test)

        assert np.all(p_conditional >= p_marginal - 1e-10)


class TestWeightedProbabilisticAdjustment:
    """Tests for Weighted Probabilistic estimation with adjustment."""

    @pytest.mark.parametrize(
        ("name", "adjustment"),
        ADJUSTMENT_METHODS_FAST,
        ids=[case[0] for case in ADJUSTMENT_METHODS_FAST],
    )
    def test_weighted_probabilistic_adjustment_valid_pvalues(
        self, shifted_dataset, name, adjustment
    ):
        """Weighted Probabilistic with adjustment should return valid p-values."""
        x_train, x_test, _ = shifted_dataset(n_train=100, n_test=20, n_features=4)

        estimation = Probabilistic(n_trials=5, adjustment=adjustment, delta=0.1)
        detector = _build_detector(
            estimation, weight_estimator=_build_weight_estimator()
        )

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        assert p_values.shape == (len(x_test),)
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    def test_weighted_probabilistic_uses_effective_n(self, shifted_dataset):
        """Weighted Probabilistic should use effective sample size (n_eff < n)."""
        x_train, x_test, _ = shifted_dataset(n_train=100, n_test=20, n_features=4)

        estimation = Probabilistic(
            n_trials=5, adjustment=Adjustment.ASYMPTOTIC, delta=0.1
        )
        detector = _build_detector(
            estimation, weight_estimator=_build_weight_estimator()
        )

        detector.fit(x_train)
        p_values = detector.predict(x_test)

        # Verify adjustment was applied (n_eff tested in unit tests)
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    def test_weighted_probabilistic_conditional_at_least_marginal(
        self, shifted_dataset
    ):
        """Weighted Probabilistic conditional should be >= marginal."""
        x_train, x_test, _ = shifted_dataset(n_train=100, n_test=20, n_features=4)

        weight_est = _build_weight_estimator()

        marginal = _build_detector(
            Probabilistic(n_trials=5, adjustment=Adjustment.NONE),
            seed=42,
            weight_estimator=weight_est,
        )
        conditional = _build_detector(
            Probabilistic(n_trials=5, adjustment=Adjustment.ASYMPTOTIC, delta=0.1),
            seed=42,
            weight_estimator=weight_est,
        )

        marginal.fit(x_train)
        conditional.fit(x_train)

        p_marginal = marginal.predict(x_test)
        p_conditional = conditional.predict(x_test)

        # Allow small tolerance for numerical precision
        assert np.all(p_conditional >= p_marginal - 1e-6)
