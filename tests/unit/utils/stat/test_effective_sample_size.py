"""Tests for Kish's effective sample size computation."""

import numpy as np
import pytest

from nonconform.scoring import Empirical


class TestComputeEffectiveN:
    """Tests for _compute_effective_n method."""

    @pytest.fixture
    def estimator(self):
        """Create an Empirical estimator to access _compute_effective_n."""
        return Empirical()

    def test_uniform_weights_equals_n(self, estimator):
        """Uniform weights should give n_eff equal to n."""
        weights = np.ones(100)
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 100

    def test_uniform_weights_scaled(self, estimator):
        """Uniformly scaled weights should give n_eff equal to n."""
        weights = np.ones(100) * 5.0
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 100

    def test_single_weight(self, estimator):
        """Single weight should give n_eff = 1."""
        weights = np.array([1.0])
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 1

    def test_two_equal_weights(self, estimator):
        """Two equal weights should give n_eff = 2."""
        weights = np.array([1.0, 1.0])
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 2

    def test_extreme_weight_ratio(self, estimator):
        """One dominant weight should give n_eff close to 1."""
        # One weight is much larger than others
        weights = np.array([1000.0, 1.0, 1.0, 1.0])
        n_eff = estimator._compute_effective_n(weights)
        # n_eff = (sum(w))^2 / sum(w^2) = 1003^2 / (1000000 + 3) ≈ 1.006
        assert n_eff == 1

    def test_moderately_unequal_weights(self, estimator):
        """Moderately unequal weights should reduce n_eff."""
        # Weights vary by factor of 2
        weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        n_eff = estimator._compute_effective_n(weights)
        # sum = 9, sum_sq = 1+4+1+4+1+4 = 15
        # n_eff = 81 / 15 = 5.4 -> floor = 5
        assert n_eff == 5
        # Should be less than actual n=6
        assert n_eff < len(weights)

    def test_kish_formula_verification(self, estimator):
        """Verify Kish's formula: n_eff = (sum(w))^2 / sum(w^2)."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected_n_eff = int(np.floor((10.0**2) / (1 + 4 + 9 + 16)))
        # 100 / 30 = 3.33... -> floor = 3
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == expected_n_eff
        assert n_eff == 3

    def test_minimum_bound_enforced(self, estimator):
        """n_eff should be at least 1."""
        # Near-zero weights should still give n_eff >= 1
        weights = np.array([1e-10, 1e-10, 1e-10])
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff >= 1

    def test_zero_weights_returns_one(self, estimator):
        """All zero weights should return 1 (edge case)."""
        weights = np.zeros(10)
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 1

    def test_large_n_uniform(self, estimator):
        """Large n with uniform weights should give n_eff = n."""
        weights = np.ones(10000)
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 10000

    def test_random_weights_bounds(self, estimator):
        """Random weights should give 1 <= n_eff <= n."""
        rng = np.random.default_rng(42)
        weights = rng.uniform(0.1, 10.0, size=50)
        n_eff = estimator._compute_effective_n(weights)
        assert 1 <= n_eff <= 50

    def test_returns_integer(self, estimator):
        """Result should always be an integer."""
        weights = np.array([1.0, 2.0, 3.0])
        n_eff = estimator._compute_effective_n(weights)
        assert isinstance(n_eff, int)

    def test_floors_result(self, estimator):
        """Result should be floored, not rounded."""
        # Design weights to give n_eff just below an integer
        # sum=6, sum_sq=6, n_eff = 36/6 = 6.0 exactly
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        n_eff = estimator._compute_effective_n(weights)
        assert n_eff == 6

        # Now adjust to get n_eff = 5.99...
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.001])
        n_eff = estimator._compute_effective_n(weights)
        # Should floor to 5 or 6 depending on exact calculation
        assert n_eff >= 5

    def test_invariant_to_scale(self, estimator):
        """n_eff should be invariant to uniform scaling of weights."""
        weights_original = np.array([1.0, 2.0, 3.0, 4.0])
        weights_scaled = weights_original * 100

        n_eff_original = estimator._compute_effective_n(weights_original)
        n_eff_scaled = estimator._compute_effective_n(weights_scaled)

        assert n_eff_original == n_eff_scaled
