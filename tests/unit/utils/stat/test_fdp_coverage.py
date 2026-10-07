"""Independent finite-sample validity checks for FDP certificates."""

from fractions import Fraction
from itertools import combinations, permutations
from math import ceil, comb, log, log1p

import numpy as np
import pytest
from scipy.optimize import brentq

from nonconform._internal import fdp_bounds as core
from nonconform.fdr import FDPCertificate


def _classical_rank_families(n_calibration: int, n_test: int) -> np.ndarray:
    # Descending pooled scores have uniformly distributed calibration/test
    # label interleavings. Each test rank counts the calibration labels above it.
    return np.array(
        [
            [
                (position - test_index + 1) / (n_calibration + 1)
                for test_index, position in enumerate(test_positions)
            ]
            for test_positions in combinations(range(n_calibration + n_test), n_test)
        ]
    )


def _exact_bernoulli_kl(value: float, target: float) -> float:
    if value == 0.0:
        return -log1p(-target)
    return value * log(value / target) + (1.0 - value) * (
        log1p(-value) - log1p(-target)
    )


def test_shared_calibration_rank_law_is_dependent_and_includes_p_value_ties():
    families = _classical_rank_families(1, 2)

    np.testing.assert_array_equal(families, [[0.5, 0.5], [0.5, 1.0], [1.0, 1.0]])
    both_minimum = np.count_nonzero(np.all(families == 0.5, axis=1))
    assert Fraction(both_minimum, len(families)) == Fraction(1, 3)
    # Independent uniform ranks would instead put probability 1/4 here.
    assert Fraction(both_minimum, len(families)) != Fraction(1, 4)


@pytest.mark.parametrize(
    ("n_calibration", "n_test", "confidence"),
    [(1, 2, 0.5), (3, 3, 0.5), (5, 4, 0.8), (10, 3, 0.5), (10, 3, 0.9)],
)
@pytest.mark.parametrize("boost", [False, True])
def test_deterministic_dkw_has_exact_classical_simultaneous_coverage(
    monkeypatch, n_calibration, n_test, confidence, boost
):
    families = _classical_rank_families(n_calibration, n_test)
    assert len(families) == comb(n_calibration + n_test, n_test)
    anchor = FDPCertificate.from_p_values(
        np.ones(n_test),
        n_calibration=n_calibration,
        confidence=confidence,
        method="ks",
        boost=boost,
    )
    envelope = anchor._envelope
    # Reuse one deterministic cutoff while enumerating all equally likely data
    # families; repeating its numerical fixed-point calculation adds no evidence.
    monkeypatch.setattr(core, "prepare_envelope", lambda **_: envelope)
    statistics = np.max(np.arange(1, n_test + 1) / n_test - families, axis=1)
    expected_covered = statistics <= envelope.summary_quantile
    observed_covered = []
    for family in families:
        certificate = FDPCertificate.from_p_values(
            family,
            n_calibration=n_calibration,
            confidence=confidence,
            method="ks",
            boost=boost,
        )
        observed_covered.append(np.all(certificate.bound_at(family) >= 1.0))

    np.testing.assert_array_equal(observed_covered, expected_covered)
    coverage = Fraction(np.count_nonzero(expected_covered), len(families))
    assert float(coverage) >= confidence


@pytest.mark.parametrize("beta", [np.nextafter(0.0, 1.0), 0.5, 1.0])
@pytest.mark.parametrize(
    "p_values",
    [
        [0.0, 0.05, 0.05],
        [0.0, 0.2, 0.2, 0.6, 0.8, 1.0],
        [0.2, 0.5, 0.7],
        [0.9, 0.95, 1.0],
    ],
)
def test_supported_thc_exponents_cover_the_entire_threshold_interval(beta, p_values):
    p_values = np.array(p_values)
    lower, upper = 0.1, 0.8
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(lower, upper, 2001),
                p_values[(p_values >= lower) & (p_values <= upper)],
            ]
        )
    )
    empirical_cdf = (p_values[:, None] <= grid).mean(axis=0)
    reference_statistic = np.max((empirical_cdf - grid) / (grid * (1.0 - grid)) ** beta)
    statistic = core._higher_criticism_statistic(
        p_values, lower=lower, upper=upper, beta=beta
    )

    assert statistic == pytest.approx(reference_statistic, rel=2e-14, abs=2e-14)
    upper_band = core._thc_ecdf_upper_bound(
        grid,
        summary_quantile=reference_statistic,
        lower=lower,
        upper=upper,
        beta=beta,
    )
    assert np.all(upper_band >= empirical_cdf - 2e-14)


@pytest.mark.parametrize("boost", [False, True])
def test_fdp_cutoff_and_boost_preserve_a_known_null_envelope_event(monkeypatch, boost):
    n_calibration, n_null = 3, 3
    alternatives = np.array([0.25, 0.8])
    n_test = n_null + len(alternatives)
    cutoff = 0.25
    monkeypatch.setattr(core, "_dkw_lambda", lambda **_: cutoff)
    covered_families = 0
    for null_p_values in _classical_rank_families(n_calibration, n_null):
        p_values = np.concatenate([null_p_values, alternatives])
        grid = np.unique(np.concatenate([[0.0, 1.0], p_values]))
        discoveries = (p_values[:, None] <= grid).sum(axis=0)
        false_discoveries = (null_p_values[:, None] <= grid).sum(axis=0)
        null_envelope = n_test * np.minimum(1.0, grid + cutoff)
        if not np.all(false_discoveries <= null_envelope):
            continue
        covered_families += 1
        actual_fdp = np.divide(
            false_discoveries,
            discoveries,
            out=np.zeros(len(grid)),
            where=discoveries > 0,
        )
        certificate = FDPCertificate.from_p_values(
            p_values, n_calibration=n_calibration, method="ks", boost=boost
        )

        assert np.all(certificate.bound_at(grid) >= actual_fdp - 2e-14)
        for threshold, count in zip(grid, discoveries, strict=True):
            assert np.count_nonzero(certificate.select(threshold)) == count
    assert 0 < covered_families < comb(n_calibration + n_null, n_null)


@pytest.mark.parametrize("precision", [1.0, 1e-3, 1e-8, 1e-100])
def test_bj_numerical_roots_are_conservative_at_coarse_and_tiny_tolerances(
    monkeypatch, precision
):
    targets = np.array([0.1, 0.25, 0.5])
    n_test, statistic = 20, 0.4
    level = statistic / n_test
    exact_roots = np.array(
        [
            brentq(
                lambda value: _exact_bernoulli_kl(value, target) - level,
                0.0,
                target,
                xtol=np.nextafter(0.0, 1.0),
                rtol=4.0 * np.finfo(float).eps,
            )
            for target in targets
        ]
    )
    original_kl = core._bernoulli_kl
    evaluations = 0

    def bounded_kl(*args):
        nonlocal evaluations
        evaluations += 1
        # Fail promptly if a regression stalls between adjacent floats.
        assert evaluations < 2000, "BJ inversion did not make numerical progress"
        return original_kl(*args)

    monkeypatch.setattr(core, "_bernoulli_kl", bounded_kl)
    roots = core._solve_bernoulli_kl_lower_bounds(
        targets, statistic, n_test=n_test, precision=precision
    )

    assert np.all(roots >= 0.0)
    assert np.all(roots <= exact_roots + 4.0 * np.finfo(float).eps)
    assert np.all(exact_roots - roots <= precision + 4.0 * np.finfo(float).eps)
    for root, target in zip(roots, targets, strict=True):
        assert _exact_bernoulli_kl(root, target) >= level - 4e-15


def test_fixed_monte_carlo_seed_has_no_conditional_nominal_coverage(monkeypatch):
    n_calibration = 900
    anchor = FDPCertificate.from_p_values(
        [1.0], n_calibration=n_calibration, confidence=0.95, seed=42
    )
    envelope = anchor._envelope
    monkeypatch.setattr(core, "prepare_envelope", lambda **_: envelope)
    covered = 0
    # One future null rank is exactly uniform over all n+1 classical values.
    for rank in range(1, n_calibration + 2):
        p_value = rank / (n_calibration + 1)
        certificate = FDPCertificate.from_p_values(
            [p_value], n_calibration=n_calibration, confidence=0.95, seed=42
        )
        covered += certificate.bound_at(p_value) >= 1.0

    assert Fraction(covered, n_calibration + 1) == Fraction(850, 901)
    assert covered / (n_calibration + 1) < anchor.confidence


@pytest.mark.parametrize(
    ("n_resamples", "confidence"), [(3, 0.1), (3, 0.5), (3, 0.95), (5, 0.8)]
)
def test_fresh_monte_carlo_joint_coverage_follows_exchangeable_order_statistics(
    n_resamples, confidence
):
    # With one randomized null p-value, the MC-KS statistic 1-p is uniform.
    # The held-out data statistic and B independent MC statistics therefore have
    # equiprobable rank permutations. This exact calculation needs no RNG draws.
    covered = 0
    total = 0
    for ranks in permutations(range(n_resamples + 1)):
        statistics = (np.array(ranks) + 1.0) / (n_resamples + 2.0)
        cutoff = core._custom_quantile(statistics[:-1], confidence)
        covered += statistics[-1] <= cutoff
        total += 1

    expected_rank = min(ceil(confidence * (n_resamples + 1)), n_resamples + 1)
    expected_coverage = Fraction(expected_rank, n_resamples + 1)
    assert Fraction(covered, total) == expected_coverage
    assert float(expected_coverage) >= confidence
