"""Unit tests for calibration-conditional p-value adjustment functions."""

import numpy as np
import pytest

from nonconform._internal.adjustment import (
    apply_adjustment,
    compute_asymptotic_sequence,
    compute_monte_carlo_sequence,
    compute_simes_sequence,
)


class TestSimesSequence:
    """Tests for compute_simes_sequence function."""

    def test_basic_output_shape(self):
        """Sequence should have length n."""
        n = 100
        b = compute_simes_sequence(n, delta=0.1)
        assert len(b) == n

    def test_monotonicity(self):
        """Sequence b_1 <= b_2 <= ... <= b_n should be non-decreasing."""
        b = compute_simes_sequence(n=100, delta=0.1)
        assert np.all(np.diff(b) >= 0), "Sequence must be non-decreasing"

    def test_bounds(self):
        """All values should be in (0, 1]."""
        b = compute_simes_sequence(n=100, delta=0.1)
        assert np.all(b > 0), "All values must be positive"
        assert np.all(b <= 1), "All values must be <= 1"

    def test_first_value_small(self):
        """First value b_1 should be small (approximately 2*log(1/delta)/n)."""
        n = 1000
        delta = 0.1
        b = compute_simes_sequence(n, delta)
        expected_approx = 2 * np.log(1 / delta) / n
        # b_1 should be close to but potentially larger than the approximation
        assert b[0] < 0.1, f"b_1 = {b[0]} should be small"
        assert b[0] > expected_approx * 0.5, "b_1 should not be too small"

    def test_last_values_near_one(self):
        """For k=n//2, values b_{n-k+2}, ..., b_n should be 1."""
        n = 100
        b = compute_simes_sequence(n, delta=0.1, k=n // 2)
        # At least some of the last values should be 1.0
        assert b[-1] == 1.0, "Last value should be 1.0"

    def test_different_delta_values(self):
        """Smaller delta should give more conservative (larger) b values."""
        n = 100
        b_small_delta = compute_simes_sequence(n, delta=0.01)
        b_large_delta = compute_simes_sequence(n, delta=0.2)
        # Smaller delta means more conservative, so b values should be larger
        assert b_small_delta[0] >= b_large_delta[0]

    def test_invalid_n(self):
        """Should raise ValueError for n < 1."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            compute_simes_sequence(n=0, delta=0.1)

    def test_invalid_delta(self):
        """Should raise ValueError for delta outside (0, 1)."""
        with pytest.raises(ValueError, match="delta must be in"):
            compute_simes_sequence(n=100, delta=0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            compute_simes_sequence(n=100, delta=1.0)

    def test_small_n(self):
        """Should work for very small n."""
        b = compute_simes_sequence(n=1, delta=0.1)
        assert len(b) == 1
        assert 0 < b[0] <= 1


class TestAsymptoticSequence:
    """Tests for compute_asymptotic_sequence function."""

    def test_basic_output_shape(self):
        """Sequence should have length n."""
        n = 100
        b = compute_asymptotic_sequence(n, delta=0.1)
        assert len(b) == n

    def test_monotonicity(self):
        """Sequence should be non-decreasing."""
        b = compute_asymptotic_sequence(n=100, delta=0.1)
        assert np.all(np.diff(b) >= 0), "Sequence must be non-decreasing"

    def test_bounds(self):
        """All values should be in (0, 1]."""
        b = compute_asymptotic_sequence(n=100, delta=0.1)
        assert np.all(b > 0), "All values must be positive"
        assert np.all(b <= 1), "All values must be <= 1"

    def test_values_above_identity(self):
        """b_i should be >= i/n (adjustment is conservative)."""
        n = 100
        b = compute_asymptotic_sequence(n, delta=0.1)
        identity = np.arange(1, n + 1) / n
        assert np.all(b >= identity - 1e-10), "b_i should be >= i/n"

    def test_large_n_tightness(self):
        """For large n, adjustment should be close to identity."""
        n = 10000
        b = compute_asymptotic_sequence(n, delta=0.1)
        identity = np.arange(1, n + 1) / n
        # For large n, the correction should be small
        max_diff = np.max(b - identity)
        assert max_diff < 0.1, f"Max difference {max_diff} should be small for large n"

    def test_invalid_parameters(self):
        """Should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError):
            compute_asymptotic_sequence(n=0, delta=0.1)
        with pytest.raises(ValueError):
            compute_asymptotic_sequence(n=100, delta=0.0)

    def test_small_n_fallback(self):
        """Should handle small n gracefully with DKW fallback."""
        b = compute_asymptotic_sequence(n=2, delta=0.1)
        assert len(b) == 2
        assert np.all(b > 0) and np.all(b <= 1)


class TestMonteCarloSequence:
    """Tests for compute_monte_carlo_sequence function."""

    def test_basic_output_shape(self):
        """Sequence should have length n."""
        n = 100
        b = compute_monte_carlo_sequence(n, delta=0.1, n_simulations=1000, seed=42)
        assert len(b) == n

    def test_monotonicity(self):
        """Sequence should be non-decreasing."""
        b = compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=1000, seed=42)
        assert np.all(np.diff(b) >= 0), "Sequence must be non-decreasing"

    def test_bounds(self):
        """All values should be in (0, 1]."""
        b = compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=1000, seed=42)
        assert np.all(b > 0), "All values must be positive"
        assert np.all(b <= 1), "All values must be <= 1"

    def test_reproducibility_with_seed(self):
        """Same seed should give same result."""
        b1 = compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=500, seed=123)
        b2 = compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=500, seed=123)
        np.testing.assert_array_equal(b1, b2)

    def test_different_seeds_give_different_results(self):
        """Different seeds should give (slightly) different results."""
        b1 = compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=500, seed=1)
        b2 = compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=500, seed=2)
        # Results may be similar but not identical
        assert not np.allclose(b1, b2, atol=1e-10)

    def test_tighter_than_simes_for_middle_values(self):
        """MC should give tighter bounds than Simes for middle p-values."""
        n = 500
        b_mc = compute_monte_carlo_sequence(n, delta=0.1, n_simulations=5000, seed=42)
        b_simes = compute_simes_sequence(n, delta=0.1)
        # For middle indices, MC should be tighter (smaller or equal)
        mid_start, mid_end = n // 4, 3 * n // 4
        assert np.all(b_mc[mid_start:mid_end] <= b_simes[mid_start:mid_end] + 1e-6), (
            "MC should be at least as tight as Simes"
        )

    def test_invalid_n_simulations(self):
        """Should raise ValueError for too few simulations."""
        with pytest.raises(ValueError, match="n_simulations must be at least"):
            compute_monte_carlo_sequence(n=100, delta=0.1, n_simulations=50)

    def test_small_n_uses_simes(self):
        """For small n, MC should fall back to Simes."""
        b_mc = compute_monte_carlo_sequence(n=5, delta=0.1, n_simulations=1000, seed=42)
        b_simes = compute_simes_sequence(n=5, delta=0.1)
        np.testing.assert_array_almost_equal(b_mc, b_simes)


class TestApplyAdjustment:
    """Tests for apply_adjustment function."""

    def test_basic_adjustment(self):
        """Adjusted p-values should be >= marginal p-values."""
        n_calib = 100
        p_marginal = np.array([0.01, 0.05, 0.10, 0.50, 0.99])
        b = compute_simes_sequence(n_calib, delta=0.1)
        p_cond = apply_adjustment(p_marginal, b, n_calib)

        assert np.all(p_cond >= p_marginal), "Conditional p-values must be >= marginal"

    def test_output_in_valid_range(self):
        """Adjusted p-values should be in [0, 1]."""
        n_calib = 100
        p_marginal = np.array([0.001, 0.01, 0.1, 0.5, 0.99, 1.0])
        b = compute_simes_sequence(n_calib, delta=0.1)
        p_cond = apply_adjustment(p_marginal, b, n_calib)

        assert np.all(p_cond >= 0), "p-values must be >= 0"
        assert np.all(p_cond <= 1), "p-values must be <= 1"

    def test_monotonicity_preserved(self):
        """If marginal p-values are sorted, conditional should be too."""
        n_calib = 100
        p_marginal = np.linspace(0.01, 1.0, 50)
        b = compute_simes_sequence(n_calib, delta=0.1)
        p_cond = apply_adjustment(p_marginal, b, n_calib)

        assert np.all(np.diff(p_cond) >= -1e-10), "Monotonicity should be preserved"

    def test_smallest_pvalue_adjustment(self):
        """Smallest possible p-value 1/(n+1) should map to b_1."""
        n_calib = 100
        p_marginal = np.array([1.0 / (n_calib + 1)])
        b = compute_simes_sequence(n_calib, delta=0.1)
        p_cond = apply_adjustment(p_marginal, b, n_calib)

        np.testing.assert_almost_equal(p_cond[0], b[0])

    def test_largest_pvalue_unchanged(self):
        """Largest p-value 1.0 should stay at 1.0."""
        n_calib = 100
        p_marginal = np.array([1.0])
        b = compute_simes_sequence(n_calib, delta=0.1)
        p_cond = apply_adjustment(p_marginal, b, n_calib)

        assert p_cond[0] == 1.0

    def test_empty_array(self):
        """Should handle empty input array."""
        n_calib = 100
        p_marginal = np.array([])
        b = compute_simes_sequence(n_calib, delta=0.1)
        p_cond = apply_adjustment(p_marginal, b, n_calib)

        assert len(p_cond) == 0


class TestSequenceComparison:
    """Tests comparing different adjustment methods."""

    def test_both_methods_give_small_b1(self):
        """Both Simes and asymptotic should give small b_1 values."""
        n = 1000
        delta = 0.1
        b_simes = compute_simes_sequence(n, delta)
        b_asym = compute_asymptotic_sequence(n, delta)

        # Both methods should give b_1 roughly around 2*log(1/delta)/n ~0.0046
        assert b_simes[0] < 0.01, f"Simes b_1 = {b_simes[0]} should be small"
        assert b_asym[0] < 0.01, f"Asymptotic b_1 = {b_asym[0]} should be small"
        # Both should be in the same ballpark
        assert abs(b_simes[0] - b_asym[0]) < 0.01

    def test_asymptotic_tighter_for_middle_values(self):
        """Asymptotic should be tighter for larger p-values."""
        n = 1000
        delta = 0.1
        b_simes = compute_simes_sequence(n, delta)
        b_asym = compute_asymptotic_sequence(n, delta)

        # For middle values, asymptotic should be smaller (tighter)
        mid = n // 2
        assert b_asym[mid] <= b_simes[mid] + 1e-6, (
            "Asymptotic should be tighter for middle values"
        )

    def test_all_methods_valid_coverage(self):
        """All methods should provide valid coverage (statistical test)."""
        n = 100
        delta = 0.1
        n_simulations = 5000
        rng = np.random.default_rng(42)

        for method_name, compute_fn in [
            ("simes", lambda: compute_simes_sequence(n, delta)),
            ("asymptotic", lambda: compute_asymptotic_sequence(n, delta)),
            (
                "monte_carlo",
                lambda: compute_monte_carlo_sequence(
                    n, delta, n_simulations=1000, seed=42
                ),
            ),
        ]:
            b = compute_fn()

            # Simulate uniform order statistics and check coverage
            violations = 0
            for _ in range(n_simulations):
                u = np.sort(rng.random(n))
                if np.any(u > b):
                    violations += 1

            coverage = 1 - violations / n_simulations
            # Allow some slack for Monte Carlo variance
            assert coverage >= 1 - delta - 0.05, (
                f"{method_name}: coverage {coverage:.3f} < {1 - delta - 0.05:.3f}"
            )
