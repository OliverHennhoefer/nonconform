"""Public false-discovery and false-alarm procedures for anomaly evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nonconform.structures import ConformalResult

from ._internal import Pruning
from ._internal import certificates as _certificates
from ._internal import e_values as _e_value_core
from ._internal import fdp_bounds as _fdp_bounds
from ._internal import fpr_bounds as _fpr_bounds
from ._internal import wcs as _wcs


@dataclass(slots=True, frozen=True)
class EValueSelectionResult:
    """Batch e-value FDR selection result.

    Attributes:
        e_values: Uniformly aggregated conformal e-values.
        selected: Boolean e-BH discovery mask.
        alpha: Target FDR level supplied to e-BH.
        alpha_bh: Fixed inner threshold used to construct split e-values.
        e_threshold: Selected e-value cutoff, or infinity when none are selected.
        n_repetitions: Number of split-conformal results aggregated.
        n_calibration: Number of calibration scores in every repetition.
        tie_seed: Seed used for randomized score ties, or None when ties were
            rejected.
    """

    e_values: np.ndarray
    selected: np.ndarray
    alpha: float
    alpha_bh: float
    e_threshold: float
    n_repetitions: int
    n_calibration: int
    tie_seed: int | None

    def __post_init__(self) -> None:
        """Copy array fields and expose them as read-only snapshots."""
        e_values = np.asarray(self.e_values, dtype=float).copy()
        selected = np.asarray(self.selected, dtype=bool).copy()
        e_values.flags.writeable = False
        selected.flags.writeable = False
        object.__setattr__(self, "e_values", e_values)
        object.__setattr__(self, "selected", selected)


@dataclass(slots=True, init=False, eq=False)
class FDPCertificate:
    """Immutable simultaneous certificate for realized FDP at p-value cutoffs.

    Construct via ``detector.fdp_bounds(x)``, ``result.fdp_bounds()``, or the
    expert ``from_p_values()`` factory. Choose the envelope method before
    inspecting its curve. Thresholds may then be explored within this fixed
    testing family. Confidence is simultaneous coverage, not an FDR target.
    Monte Carlo coverage is joint over the data and independent envelope
    sampling, not conditional on every fixed seed or realized envelope. Choose
    settings before inspecting bounds; do not cherry-pick seeds or repeated
    constructions. The deterministic ``ks`` method has no sampling randomness.

    Evidence and default-grid diagnostics are read-only arrays. Queries never
    resample. ``select(t)`` returns an original-order NumPy mask for p <= t;
    t is a p-value cutoff, not a requested FDP bound.
    """

    _p_values: np.ndarray = field(repr=False)
    _support: np.ndarray = field(repr=False)
    _counts: np.ndarray = field(repr=False)
    _prefix_minima: np.ndarray | None = field(repr=False)
    _envelope: _fdp_bounds.Envelope = field(repr=False)

    def __init__(self) -> None:
        """Require validated construction through the certificate factories."""
        raise TypeError(
            "Use detector.fdp_bounds(), result.fdp_bounds(), or "
            "FDPCertificate.from_p_values()."
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent mutation after validated factory construction."""
        raise AttributeError("FDPCertificate is immutable.")

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of certificate state."""
        raise AttributeError("FDPCertificate is immutable.")

    def __repr__(self) -> str:
        """Summarize configuration without dumping evidence arrays."""
        return (
            f"FDPCertificate(method={self.method!r}, confidence={self.confidence}, "
            f"n_test={self.n_test}, n_calibration={self.n_calibration}, "
            f"boost={self.boost})"
        )

    @classmethod
    def from_p_values(
        cls,
        p_values: np.ndarray,
        *,
        n_calibration: int,
        confidence: float = 0.95,
        method: str = "mc_thc",
        n_resamples: int | None = None,
        seed: int | None = None,
        boost: bool = True,
        lower: float | None = None,
        upper: float | None = None,
        beta: float | None = None,
    ) -> FDPCertificate:
        """Certify external p-values; the caller owns provenance assumptions.

        Requires unweighted empirical split-conformal p-values from a fixed
        scoring map and the reference method's exchangeability assumptions.
        Native detector/snapshot entry points check supported scope; this expert
        route cannot. Scientific exchangeability is never established by code.

        Args:
            p_values: Nonempty 1D testing family in [0, 1], in original order.
            n_calibration: Positive calibration size shared by all p-values.
            confidence: Simultaneous coverage probability in (0, 1), jointly
                over data and independent sampling for Monte Carlo methods.
            method: mc_thc (default), mc_hc, mc_ks, ks, or mc_bj.
            n_resamples: Monte Carlo draws; defaults to 1000 for MC methods.
            seed: Monte Carlo seed only. None draws fresh randomness once.
            boost: Apply threshold-specific sharpening (default True).
            lower: THC lower truncation, default 0.01.
            upper: THC upper truncation, default 0.99.
            beta: THC exponent in (0, 1], default 0.5.

        Method-specific options must be omitted or None when inapplicable.
        Deterministic ks accepts neither n_resamples nor seed. A fixed Monte
        Carlo seed gives reproducibility, not a separate conditional coverage
        guarantee. Choose all settings before inspecting the curve; do not
        cherry-pick seeds or reconstruct certificates to improve their bounds.
        Berk-Jones inversion uses a conservative internal numerical tolerance.

        References:
            Song, Jin, and Candès, "Everywhere Valid Bounds on False Discovery
            Proportions in Conformal Inference" (2026), arXiv:2605.20726.
        """
        values = _fdp_bounds.as_p_values("p_values", p_values)
        envelope = _fdp_bounds.prepare_envelope(
            n_calibration=n_calibration,
            n_test=values.size,
            confidence=confidence,
            method=method,
            n_resamples=n_resamples,
            seed=seed,
            boost=boost,
            lower=lower,
            upper=upper,
            beta=beta,
        )
        support, counts = np.unique(values, return_counts=True)
        counts = np.cumsum(counts)
        minima = None
        if boost:
            minima = _certificates.immutable_array(
                np.minimum.accumulate(
                    np.minimum(
                        values.size, values.size * envelope.evaluate(support) - counts
                    )
                )
            )
        certificate = object.__new__(cls)
        object.__setattr__(
            certificate, "_p_values", _certificates.immutable_array(values)
        )
        object.__setattr__(
            certificate, "_support", _certificates.immutable_array(support)
        )
        object.__setattr__(
            certificate, "_counts", _certificates.immutable_array(counts)
        )
        object.__setattr__(certificate, "_prefix_minima", minima)
        object.__setattr__(certificate, "_envelope", envelope)
        return certificate

    def _query(self, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return counts and bounds using prepared inclusive support ranks."""
        positions = np.searchsorted(self._support, thresholds, side="right") - 1
        counts = np.where(positions >= 0, self._counts[positions], 0)
        if self._prefix_minima is not None:
            numerator = self._prefix_minima[positions] + counts
        else:
            numerator = self.n_test * self._envelope.evaluate(thresholds)
        bounds = np.divide(
            numerator,
            counts,
            out=np.zeros_like(numerator, dtype=float),
            where=counts > 0,
        )
        return counts, np.clip(bounds, 0.0, 1.0)

    def bound_at(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the simultaneous FDP bound at scalar or vector cutoffs."""
        thresholds, scalar = _fdp_bounds.as_threshold_query(threshold)
        _, bounds = self._query(thresholds)
        return float(bounds[0]) if scalar else bounds

    def precision_at(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Return 1 - bound_at(threshold), a simultaneous precision lower bound."""
        return 1.0 - self.bound_at(threshold)

    def select(self, threshold: float) -> np.ndarray:
        """Return an original-order Boolean NumPy mask for p <= threshold."""
        thresholds, scalar = _fdp_bounds.as_threshold_query(threshold)
        if not scalar:
            raise ValueError("threshold must be a scalar for select().")
        return self._p_values <= thresholds[0]

    def to_frame(self, thresholds: np.ndarray | None = None) -> pd.DataFrame:
        """Report a grid; default to sorted unique observed p-values.

        Explicit grids preserve order and duplicates and may be empty.
        Returned tables are independent of the certificate.
        """
        grid = (
            self._support
            if thresholds is None
            else _fdp_bounds.as_thresholds(thresholds)
        )
        counts, bounds = self._query(grid)
        return pd.DataFrame(
            {
                "threshold": grid.copy(),
                "discoveries": counts,
                "fdp_upper_bound": bounds,
                "precision_lower_bound": 1.0 - bounds,
            }
        )

    @property
    def p_values(self) -> np.ndarray:
        """Read-only evidence in original observation order."""
        return _certificates.immutable_array(self._p_values)

    @property
    def thresholds(self) -> np.ndarray:
        """Read-only default grid of sorted unique observed p-values."""
        return _certificates.immutable_array(self._support)

    @property
    def rejection_counts(self) -> np.ndarray:
        """Read-only discovery counts on the default grid."""
        return _certificates.immutable_array(self._counts)

    @property
    def fdp_upper_bounds(self) -> np.ndarray:
        """Read-only FDP upper bounds on the default grid."""
        return _certificates.immutable_array(self.bound_at(self._support))

    @property
    def precision_lower_bounds(self) -> np.ndarray:
        """Read-only precision lower bounds on the default grid."""
        return _certificates.immutable_array(self.precision_at(self._support))

    @property
    def n_calibration(self) -> int:
        """Calibration sample size."""
        return self._envelope.n_calibration

    @property
    def n_test(self) -> int:
        """Fixed testing-family size."""
        return self._envelope.n_test

    @property
    def confidence(self) -> float:
        """Coverage level, joint over data and sampling for Monte Carlo methods."""
        return self._envelope.confidence

    @property
    def method(self) -> str:
        """Normalized envelope method."""
        return self._envelope.method

    @property
    def n_resamples(self) -> int | None:
        """Effective Monte Carlo draws; None for deterministic KS."""
        return self._envelope.n_resamples

    @property
    def seed(self) -> int | None:
        """Monte Carlo seed supplied at construction, or None."""
        return self._envelope.seed

    @property
    def boost(self) -> bool:
        """Whether threshold-specific sharpening is enabled."""
        return self._envelope.boost

    @property
    def lower(self) -> float | None:
        """Effective THC lower truncation; otherwise None."""
        return self._envelope.lower

    @property
    def upper(self) -> float | None:
        """Effective THC upper truncation; otherwise None."""
        return self._envelope.upper

    @property
    def beta(self) -> float | None:
        """Effective THC exponent; otherwise None."""
        return self._envelope.beta


@dataclass(slots=True, init=False, eq=False)
class FPRCertificate:
    """Immutable simultaneous certificate for raw-score false-positive rates.

    Construct via ``detector.fpr_bounds()``, ``result.fpr_bounds()``, or the
    expert ``from_scores()`` factory. Calibration scores must be finite and use
    the anomalous-higher convention. The certificate's uniform upper band can
    be queried at arbitrary score thresholds and inverted to choose a threshold
    for a target false-positive rate.

    Coverage requires clean calibration and future inlier scores that are
    i.i.d. conditional on a scoring map fixed independently of calibration.
    Exchangeability alone is insufficient.

    With probability at least ``confidence`` over the calibration draw,
    conditional on the independently fitted scoring map, the upper band covers
    the population inlier false-alarm rate simultaneously at every threshold.
    On that event, a threshold selected for ``target_fpr`` has future-inlier
    false-alarm probability at most that target. The deterministic finite-sample
    one-sided KS band is prepared once and reused by all threshold queries.
    Copying and pickle roundtrips preserve this band and immutable evidence.
    """

    _calibration_scores: np.ndarray = field(repr=False)
    _thresholds: np.ndarray = field(repr=False)
    _alarm_counts: np.ndarray = field(repr=False)
    _band: _fpr_bounds.KSBand = field(repr=False)

    def __init__(self) -> None:
        """Require validated construction through the certificate factories."""
        raise TypeError(
            "Use detector.fpr_bounds(), result.fpr_bounds(), or "
            "FPRCertificate.from_scores()."
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent mutation after validated factory construction."""
        raise AttributeError("FPRCertificate is immutable.")

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of certificate state."""
        raise AttributeError("FPRCertificate is immutable.")

    def __repr__(self) -> str:
        """Summarize configuration without dumping calibration scores."""
        return (
            f"FPRCertificate(method={self.method!r}, "
            f"confidence={self.confidence}, "
            f"n_calibration={self.n_calibration})"
        )

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[np.ndarray, _fpr_bounds.KSBand], FPRCertificate],
        tuple[np.ndarray, _fpr_bounds.KSBand],
    ]:
        """Rebuild immutable evidence while retaining the prepared KS band."""
        return self._from_calibration, (self._calibration_scores, self._band)

    @classmethod
    def from_scores(
        cls,
        calibration_scores: np.ndarray,
        *,
        confidence: float = 0.95,
    ) -> FPRCertificate:
        """Certify an external anomalous-higher calibration-score sample.

        This expert route trusts the caller to provide clean calibration scores
        that are i.i.d. from the deployment inlier distribution, conditional on
        a scoring map fixed independently of calibration. Future inlier scores
        must be independent draws from that same distribution. Exchangeability
        alone is insufficient. It constructs a deterministic finite-sample
        one-sided KS band for the false-positive rate of every raw-score
        threshold. Confidence is simultaneous coverage over the calibration
        draw, conditional on the independently fitted scoring map.

        Args:
            calibration_scores: Nonempty, finite, one-dimensional scores for
                calibration observations known to represent inliers.
            confidence: Simultaneous coverage probability in ``(0, 1)``.

        Returns:
            An immutable raw-score FPR certificate.

        Raises:
            RuntimeError: If the numerical KS cutoff is not finite or lies
                outside ``[0, 1]``.

        Note:
            The score convention is anomalous-higher. Native detector entry
            points normalize detector polarity before constructing the
            certificate; external callers must do so themselves.
        """
        scores = _fpr_bounds.as_calibration_scores(
            "calibration_scores", calibration_scores
        )
        band = _fpr_bounds.prepare_ks_band(
            n_calibration=scores.size,
            confidence=confidence,
        )
        return cls._from_calibration(scores, band)

    @classmethod
    def _from_calibration(
        cls, scores: np.ndarray, band: _fpr_bounds.KSBand
    ) -> FPRCertificate:
        """Own validated evidence and derive its complete decision grid."""
        sorted_scores = np.sort(scores)
        with np.errstate(over="ignore"):
            thresholds = np.unique(
                np.r_[-np.inf, np.nextafter(sorted_scores, np.inf), np.inf]
            )
        alarm_counts = scores.size - np.searchsorted(
            sorted_scores, thresholds, side="left"
        )

        certificate = object.__new__(cls)
        object.__setattr__(
            certificate,
            "_calibration_scores",
            _certificates.immutable_array(scores),
        )
        object.__setattr__(
            certificate,
            "_thresholds",
            _certificates.immutable_array(thresholds),
        )
        object.__setattr__(
            certificate,
            "_alarm_counts",
            _certificates.immutable_array(alarm_counts),
        )
        object.__setattr__(certificate, "_band", band)
        return certificate

    def _query(self, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return inclusive calibration tail counts and upper bounds."""
        positions = np.searchsorted(self._thresholds, thresholds, side="right") - 1
        counts = self._alarm_counts[positions]
        bounds = _fpr_bounds.upper_bound(
            counts,
            thresholds=thresholds,
            n_calibration=self.n_calibration,
            critical_value=self.critical_value,
        )
        return counts, bounds

    def bound_at(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the simultaneous FPR upper bound at score thresholds."""
        thresholds, scalar = _fpr_bounds.as_threshold_query(threshold)
        _, bounds = self._query(thresholds)
        return float(bounds[0]) if scalar else bounds

    def threshold_for(self, target_fpr: float) -> np.float64:
        """Return the least-conservative threshold certified for ``target_fpr``.

        The returned threshold uses the inclusive ``score >= threshold`` rule.
        It is a ``numpy.float64`` scalar to preserve precision when comparing
        with lower-precision NumPy scores. Keep this scalar dtype for direct
        comparisons, or use ``select()``.
        If no finite threshold supported by the calibration scores can satisfy
        the target, ``numpy.inf`` is returned; applying it to finite scores
        produces an empty selection with certified FPR zero.
        """
        target = _fpr_bounds.validate_target_fpr(target_fpr)
        _, bounds = self._query(self._thresholds)
        return self._thresholds[np.flatnonzero(bounds <= target)[0]]

    def select(
        self,
        scores: np.ndarray,
        *,
        threshold: float | None = None,
        target_fpr: float | None = None,
    ) -> np.ndarray:
        """Return an original-order mask for an explicit or certified threshold.

        Exactly one of ``threshold`` or ``target_fpr`` must be supplied. Scores
        must use the anomalous-higher convention and are classified with the
        inclusive rule ``score >= threshold``.
        """
        if (threshold is None) == (target_fpr is None):
            raise ValueError("Supply exactly one of threshold or target_fpr.")
        if target_fpr is not None:
            threshold_value = self.threshold_for(target_fpr)
        else:
            thresholds, scalar = _fpr_bounds.as_threshold_query(threshold)
            if not scalar:
                raise ValueError("threshold must be a scalar for select().")
            threshold_value = float(thresholds[0])
        return _fpr_bounds.as_scores("scores", scores) >= threshold_value

    def to_frame(self, thresholds: np.ndarray | None = None) -> pd.DataFrame:
        """Report empirical and certified FPR on a threshold grid.

        The default grid starts at ``-inf``, steps just above each unique
        calibration score, and ends at ``inf`` for the empty rule. It includes
        every threshold considered by ``threshold_for()``.
        Explicit grids preserve their order and duplicates.
        """
        if thresholds is None:
            grid = self._thresholds
        else:
            grid, scalar = _fpr_bounds.as_threshold_query(thresholds)
            if scalar:
                raise ValueError("thresholds must be a 1D array.")
        counts, bounds = self._query(grid)
        return pd.DataFrame(
            {
                "threshold": grid.copy(),
                "alarm_counts": counts,
                "empirical_fpr": counts / self.n_calibration,
                "fpr_upper_bound": bounds,
            }
        )

    @property
    def calibration_scores(self) -> np.ndarray:
        """Read-only calibration scores in their original order."""
        return _certificates.immutable_array(self._calibration_scores)

    @property
    def thresholds(self) -> np.ndarray:
        """Read-only complete decision grid, including both infinite endpoints."""
        return _certificates.immutable_array(self._thresholds)

    @property
    def alarm_counts(self) -> np.ndarray:
        """Read-only inclusive calibration alarm counts on the default grid."""
        return _certificates.immutable_array(self._alarm_counts)

    @property
    def empirical_fpr(self) -> np.ndarray:
        """Empirical calibration FPR on the default threshold grid."""
        return _certificates.immutable_array(
            self._alarm_counts.astype(float) / self.n_calibration
        )

    @property
    def fpr_upper_bounds(self) -> np.ndarray:
        """Simultaneous FPR upper bounds on the default threshold grid."""
        return _certificates.immutable_array(self.bound_at(self._thresholds))

    @property
    def critical_value(self) -> float:
        """Prepared finite-sample one-sided KS critical value and band floor."""
        return self._band.critical_value

    @property
    def n_calibration(self) -> int:
        """Calibration sample size."""
        return self._band.n_calibration

    @property
    def confidence(self) -> float:
        """Simultaneous coverage probability over the clean calibration draw."""
        return self._band.confidence

    @property
    def method(self) -> str:
        """The fixed certificate construction method."""
        return "ks"


def conformal_e_values(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha_bh: float,
    tie_seed: int | None = None,
) -> np.ndarray:
    """Compute derandomized conformal e-values from split-conformal scores.

    This low-level array interface trusts the caller to provide repetitions for
    the same test family in the same observation order. Repetitions are
    aggregated uniformly.

    Args:
        test_scores: Test anomaly scores. Shape ``(n_test,)`` for one split or
            ``(n_repetitions, n_test)`` for repeated splits.
        calib_scores: Calibration anomaly scores with matching split dimension.
        alpha_bh: Inner BH-style threshold for each split construction.
        tie_seed: ``None`` rejects tied scores. A non-negative integer
            reproducibly randomizes unique secondary ranks for ties.

    Returns:
        Aggregated e-values of shape ``(n_test,)``.

    Raises:
        TypeError: If ``tie_seed`` has an unsupported type.
        ValueError: If score inputs, ``alpha_bh``, or ``tie_seed`` are invalid,
            or tied scores are found when ``tie_seed`` is None.
    """
    return _e_value_core.compute_conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=alpha_bh,
        tie_seed=tie_seed,
    )


def e_value_false_discovery_control(
    e_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply the e-BH procedure to non-negative e-values.

    Args:
        e_values: Non-negative e-values; larger values are stronger evidence.
        alpha: Target FDR level in ``(0, 1)``.

    Returns:
        Boolean selection mask aligned with ``e_values``.
    """
    alpha_value = _fdp_bounds.validate_probability("alpha", alpha)
    e_values_arr = _e_value_core.normalize_e_values(e_values)
    selected, _ = _e_value_core.e_bh_selection(e_values_arr, alpha=alpha_value)
    return selected


def select_conformal_e_values(
    results: Sequence[ConformalResult],
    *,
    alpha: float = 0.05,
    alpha_bh: float | None = None,
    tie_seed: int | None = None,
) -> EValueSelectionResult:
    """Select a fixed test family from repeated split-conformal results.

    Native detector provenance checks integrated, unweighted ``Split`` results
    and the recorded test-batch content and ordering. Supply unmodified snapshots:
    changes to their score arrays are not tracked. Manual or unstamped results
    are unsupported; expert callers can use :func:`conformal_e_values` directly.

    Args:
        results: Non-empty sequence of unmodified detector-produced snapshots.
        alpha: Target FDR level for the final e-BH procedure.
        alpha_bh: Inner threshold, defaulting to ``alpha / 10``.
        tie_seed: ``None`` rejects ties. A non-negative integer reproducibly
            randomizes unique secondary ranks for ties.

    Returns:
        Aggregated e-values, final e-BH mask, and diagnostics.

    Raises:
        TypeError: If ``results`` or ``tie_seed`` has an unsupported type.
        ValueError: If provenance, batch identity, score arrays, probabilities,
            or tied-score handling are unsupported.
    """
    alpha_value = _fdp_bounds.validate_probability("alpha", alpha)
    test_scores, calib_scores = _e_value_core.scores_from_results(results)
    return _select_conformal_e_values_from_scores(
        test_scores,
        calib_scores,
        alpha=alpha_value,
        alpha_bh=alpha_bh,
        tie_seed=tie_seed,
    )


def _select_conformal_e_values_from_scores(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha: float,
    alpha_bh: float | None,
    tie_seed: int | None,
) -> EValueSelectionResult:
    """Select from repetition matrices whose construction scope is verified."""
    alpha_value = _fdp_bounds.validate_probability("alpha", alpha)
    alpha_bh_value = alpha_value / 10.0 if alpha_bh is None else alpha_bh
    e_values = _e_value_core.compute_conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=alpha_bh_value,
        tie_seed=tie_seed,
    )
    selected, e_threshold = _e_value_core.e_bh_selection(
        e_values,
        alpha=alpha_value,
    )
    return EValueSelectionResult(
        e_values=e_values,
        selected=selected,
        alpha=alpha_value,
        alpha_bh=float(alpha_bh_value),
        e_threshold=e_threshold,
        n_repetitions=int(test_scores.shape[0]),
        n_calibration=int(calib_scores.shape[1]),
        tie_seed=tie_seed,
    )


def weighted_false_discovery_control(
    result: ConformalResult | None,
    *,
    alpha: float = 0.05,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Apply weighted conformalized selection to a result bundle.

    The result must contain p-values, test and calibration scores, and matching
    non-negative weights for the same complete testing family. Validity also
    depends on the weighted-conformal covariate-shift assumptions.

    Args:
        result: Weighted detector result for the target family.
        alpha: Nominal FDR target in ``(0, 1)``.
        pruning: Deterministic, homogeneous-randomized, or
            heterogeneous-randomized WCS pruning rule.
        seed: Non-negative seed for randomized pruning, or None.

    Returns:
        Boolean selection mask aligned with the result's test rows.
    """
    p_values, test_scores, calib_scores, test_weights, calib_weights = (
        _wcs.extract_required_fields(result)
    )
    kde_support, use_self_weight = _wcs.extract_kde_support(result)
    return _wcs.run(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
        pruning=pruning,
        seed=seed,
        kde_support=kde_support,
        include_self_weight=use_self_weight,
    )


def weighted_false_discovery_control_from_arrays(
    *,
    p_values: np.ndarray,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float = 0.05,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Apply weighted conformalized selection to explicit arrays.

    This low-level API cannot verify provenance. All arrays must come from the
    same calibration construction and complete target family.

    Args:
        p_values: One p-value per test observation.
        test_scores: One anomalous-higher score per test observation.
        calib_scores: Calibration scores in the same orientation.
        test_weights: Non-negative target-density weights for test observations.
        calib_weights: Non-negative target-density weights for calibration
            observations.
        alpha: Nominal FDR target in ``(0, 1)``.
        pruning: WCS pruning rule.
        seed: Non-negative seed for randomized pruning, or None.

    Returns:
        Boolean selection mask aligned with the test arrays.
    """
    return _wcs.run(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
        pruning=pruning,
        seed=seed,
    )


__all__ = [
    "EValueSelectionResult",
    "FDPCertificate",
    "FPRCertificate",
    "Pruning",
    "conformal_e_values",
    "e_value_false_discovery_control",
    "select_conformal_e_values",
    "weighted_false_discovery_control",
    "weighted_false_discovery_control_from_arrays",
]
