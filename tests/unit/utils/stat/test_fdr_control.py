"""Unit tests for FDR control verification.

Based on the investigation in tests/investigation/test_fdr_empirical_verification.py.
These tests verify that the WCS algorithm maintains valid FDR control under various
configurations using Monte Carlo simulation.
"""

import numpy as np
import pytest

from nonconform._internal import Pruning
from nonconform.fdr import weighted_false_discovery_control


def _run_fdr_simulation(
    n_trials: int,
    n_calib: int,
    n_test: int,
    n_anomalies: int,
    anomaly_shift: float,
    alpha: float,
    pruning: Pruning,
    seed: int,
    use_weights: bool,
) -> dict:
    """Run FDR simulation with synthetic data."""
    rng = np.random.default_rng(seed)
    fdp_list = []
    power_list = []

    for trial in range(n_trials):
        # Generate calibration scores (all normal)
        calib_scores = rng.standard_normal(n_calib)

        # Generate test scores (mix of normal and anomalies)
        n_normal = n_test - n_anomalies
        test_normal = rng.standard_normal(n_normal)
        test_anomalies = rng.standard_normal(n_anomalies) + anomaly_shift
        test_scores = np.concatenate([test_normal, test_anomalies])

        # True labels: 0 = normal (null), 1 = anomaly (alternative)
        true_labels = np.array([0] * n_normal + [1] * n_anomalies)

        # Shuffle test set
        shuffle_idx = rng.permutation(n_test)
        test_scores = test_scores[shuffle_idx]
        true_labels = true_labels[shuffle_idx]

        # Generate weights
        if use_weights:
            calib_weights = np.exp(0.1 * np.abs(calib_scores))
            test_weights = np.exp(0.1 * np.abs(test_scores))
            calib_weights = calib_weights / np.mean(calib_weights)
            test_weights = test_weights / np.mean(test_weights)
        else:
            calib_weights = np.ones(n_calib)
            test_weights = np.ones(n_test)

        # Apply WCS
        discoveries_mask = weighted_false_discovery_control(
            p_values=None,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=alpha,
            pruning=pruning,
            seed=trial,
        )

        # Compute FDP and power
        discoveries = np.where(discoveries_mask)[0]
        n_discoveries = len(discoveries)

        if n_discoveries > 0:
            false_discoveries = np.sum(true_labels[discoveries] == 0)
            fdp = false_discoveries / n_discoveries
            true_discoveries = np.sum(true_labels[discoveries] == 1)
            power = true_discoveries / n_anomalies if n_anomalies > 0 else 0
        else:
            fdp = 0.0
            power = 0.0

        fdp_list.append(fdp)
        power_list.append(power)

    return {
        "empirical_fdr": np.mean(fdp_list),
        "fdr_std": np.std(fdp_list) / np.sqrt(len(fdp_list)),
        "empirical_power": np.mean(power_list),
    }


class TestFDRControl:
    """Test that WCS maintains FDR control under various configurations."""

    @pytest.mark.parametrize(
        "alpha,pruning",
        [
            (0.05, Pruning.DETERMINISTIC),
            (0.10, Pruning.DETERMINISTIC),
            (0.20, Pruning.DETERMINISTIC),
            (0.10, Pruning.HOMOGENEOUS),
            (0.10, Pruning.HETEROGENEOUS),
        ],
    )
    def test_fdr_controlled(self, alpha, pruning):
        """Test FDR is controlled at target level."""
        result = _run_fdr_simulation(
            n_trials=200,
            n_calib=500,
            n_test=100,
            n_anomalies=10,
            anomaly_shift=3.0,
            alpha=alpha,
            pruning=pruning,
            seed=42,
            use_weights=True,
        )

        # Allow 3 standard errors margin for statistical tolerance
        margin = 3 * result["fdr_std"]
        assert result["empirical_fdr"] <= alpha + margin, (
            f"FDR {result['empirical_fdr']:.4f} exceeds "
            f"target {alpha} + margin {margin:.4f}"
        )

    def test_power_nonzero(self):
        """Test that power is non-zero with strong signal."""
        result = _run_fdr_simulation(
            n_trials=200,
            n_calib=500,
            n_test=100,
            n_anomalies=10,
            anomaly_shift=3.0,
            alpha=0.10,
            pruning=Pruning.DETERMINISTIC,
            seed=42,
            use_weights=True,
        )

        # With a strong signal (shift=3.0), power should be substantial
        assert result["empirical_power"] > 0.3, (
            f"Power {result['empirical_power']:.4f} is too low for strong signal"
        )

    def test_unweighted_fdr_controlled(self):
        """Test FDR control with uniform weights."""
        result = _run_fdr_simulation(
            n_trials=200,
            n_calib=500,
            n_test=100,
            n_anomalies=10,
            anomaly_shift=3.0,
            alpha=0.10,
            pruning=Pruning.DETERMINISTIC,
            seed=42,
            use_weights=False,
        )

        margin = 3 * result["fdr_std"]
        assert result["empirical_fdr"] <= 0.10 + margin
