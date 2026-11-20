"""Core data structures and protocols for nonconform.

This module provides the fundamental types used throughout the package:

Classes:
    AnomalyDetector: Protocol defining the detector interface.
    ConformalResult: Container for conformal inference outputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nonconform._internal.provenance import ResultProvenance
    from nonconform.fdr import FDPCertificate, FPRCertificate


def _array_summary(arr: np.ndarray | None) -> str:
    """Return compact summary for ndarray fields."""
    if arr is None:
        return "None"
    return f"array(shape={arr.shape}, dtype={arr.dtype})"


@runtime_checkable
class AnomalyDetector(Protocol):
    """Protocol defining the interface for anomaly detectors.

    A supported PyOD model, a recognized scikit-learn estimator, or a custom
    object can be used with nonconform when it provides this interface. The
    object must also support shallow and deep copying because resampling
    strategies create independent detector replicas.

    Required methods:
        fit: Train the detector on data
        decision_function: Compute anomaly scores
        get_params: Retrieve detector parameters
        set_params: Configure detector parameters

    Examples:
        ```python
        from sklearn.ensemble import IsolationForest

        from nonconform.structures import AnomalyDetector

        detector: AnomalyDetector = IsolationForest(random_state=42)
        print(isinstance(detector, AnomalyDetector))
        ```
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the anomaly detector.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            The fitted detector instance.
        """
        ...

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Score direction is detector-specific. Pass the corresponding
        ``score_polarity`` to :class:`~nonconform.detector.ConformalDetector`
        when it cannot be inferred safely.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this detector.

        Args:
            deep: If True, return parameters for sub-objects.

        Returns:
            Parameter names mapped to their values.
        """
        ...

    def set_params(self, **params: Any) -> Self:
        """Set parameters for this detector.

        Args:
            **params: Detector parameters.

        Returns:
            The detector instance.
        """
        ...


@dataclass(slots=True)
class ConformalResult:
    """Snapshot of detector outputs for downstream procedures.

    This dataclass holds the latest p-values or score-tail estimates, raw scores,
    and optional weights produced by a detector call.

    Attributes:
        p_values: Values produced by the configured estimation strategy, or None
            when only scores were requested. With ``Empirical``, these are
            rank-based conformal p-values.
        test_scores: Aggregated, anomalous-higher scores for test instances.
        calib_scores: Anomalous-higher scores for the calibration set, or None
            for aggregated raw DerandomizedSplits scores, which have no single
            corresponding calibration distribution.
        test_weights: Importance weights for test instances (weighted mode only).
        calib_weights: Importance weights for calibration instances.
        metadata: Method metadata, including the strategy, estimator, and
            weighted-mode marker for p-value computations.

    Examples:
        ```python
        import numpy as np

        from nonconform.structures import ConformalResult

        result = ConformalResult(
            p_values=np.array([0.50, 0.02]),
            test_scores=np.array([0.1, 2.4]),
            calib_scores=np.array([-0.2, 0.0, 0.3, 0.8]),
            metadata={"nonconform": {"weighted": False}},
        )
        print(result.p_values)
        print(result.metadata)
        ```
    """

    p_values: np.ndarray | None = None
    test_scores: np.ndarray | None = None
    calib_scores: np.ndarray | None = None
    test_weights: np.ndarray | None = None
    calib_weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _provenance: ResultProvenance | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    # Index of the pandas test batch that produced this snapshot, if any.
    _index: pd.Index | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        """Return concise notebook-friendly result summary."""
        metadata_keys = sorted(self.metadata.keys())
        if len(metadata_keys) > 5:
            metadata_repr = f"{metadata_keys[:5]}... (total={len(metadata_keys)})"
        else:
            metadata_repr = str(metadata_keys)
        return (
            "ConformalResult("
            f"p_values={_array_summary(self.p_values)}, "
            f"test_scores={_array_summary(self.test_scores)}, "
            f"calib_scores={_array_summary(self.calib_scores)}, "
            f"test_weights={_array_summary(self.test_weights)}, "
            f"calib_weights={_array_summary(self.calib_weights)}, "
            f"metadata_keys={metadata_repr})"
        )

    def adjusted_p_values(
        self,
        *,
        method: Literal["bh", "storey"] = "bh",
    ) -> np.ndarray:
        """Return multiplicity-adjusted p-values for this snapshot's family.

        For the snapshot recorded by an unweighted ``select(x, alpha=a,
        method=m)`` call, ``adjusted_p_values(method=m) <= a`` reproduces the
        returned mask. The values are in original observation order and treat
        the complete snapshot as one testing family. Choose the FDR target
        before inspecting them.

        ``"bh"`` returns Benjamini-Hochberg adjusted p-values. ``"storey"``
        multiplies them by Storey's null-proportion estimate and caps them at
        one; it requires classical conformal p-values from ``Empirical`` or
        ``ConditionalEmpirical`` with ``tie_break="classical"``.

        Requires an unmodified native snapshot of unweighted p-values. Weighted
        selection uses WCS, which has no adjusted p-values. For external
        p-values, use ``scipy.stats.false_discovery_control``.

        Args:
            method: ``"bh"`` or ``"storey"``. Defaults to ``"bh"``.

        Returns:
            A new array of adjusted p-values in ``[0, 1]``.

        Raises:
            ValueError: If the snapshot has no p-values, is weighted, lacks
                native provenance, or is outside the Storey scope.

        Examples:
            ```python
            import numpy as np
            from sklearn.ensemble import IsolationForest

            from nonconform import ConformalDetector, Split

            rng = np.random.default_rng(1)
            x_reference = rng.normal(size=(1_000, 2))
            x_test = np.vstack(
                [rng.normal(size=(40, 2)), rng.normal(loc=5.0, size=(10, 2))]
            )
            detector = ConformalDetector(
                detector=IsolationForest(random_state=1),
                strategy=Split(n_calib=0.5),
                seed=1,
            ).fit(x_reference)
            detector.compute_p_values(x_test)
            adjusted = detector.last_result.adjusted_p_values(method="storey")
            print(np.flatnonzero(adjusted <= 0.1))
            ```
        """
        from nonconform._internal.p_value_selection import (
            adjusted_p_values,
            normalize_method,
            validate_result_scope,
        )

        normalized = normalize_method(method)
        p_values, n_calibration = validate_result_scope(self, method=normalized)
        return adjusted_p_values(
            p_values, method=normalized, n_calibration=n_calibration
        )

    def fdp_bounds(self, *, confidence: float = 0.95) -> FDPCertificate:
        """Certify this snapshot without rescoring.

        Requires an unmodified native snapshot of unweighted empirical Split
        inference, including detached calibration. Scope and batch dimensions
        are checked; these checks do not prove array integrity or scientific
        exchangeability. For external p-values or advanced envelope settings,
        use FDPCertificate.from_p_values and verify its assumptions.

        Uses 1000 fresh Monte Carlo draws with the default truncated
        higher-criticism envelope and threshold-specific sharpening. Confidence
        is simultaneous coverage jointly over data and independent envelope
        sampling, not an FDR target or a guarantee conditional on every realized
        envelope. Do not repeatedly construct certificates to cherry-pick bounds.
        Queries on the returned immutable certificate never resample. Later
        edits to this snapshot cannot affect it.
        """
        from nonconform._internal.fdp_bounds import validate_result_scope
        from nonconform.fdr import FDPCertificate

        n_calibration = validate_result_scope(self)
        return FDPCertificate.from_p_values(
            self.p_values,
            n_calibration=n_calibration,
            confidence=confidence,
        )

    def fpr_bounds(
        self,
        *,
        confidence: float = 0.95,
    ) -> FPRCertificate:
        """Certify the raw-score false-positive rate without rescoring.

        Requires an unmodified native snapshot of unweighted Split inference,
        including detached calibration, independently of the p-value estimator.
        Calibration and future inlier scores must be i.i.d. conditional on a
        scoring map fixed independently of calibration; exchangeability alone
        is insufficient.
        Scope checks do not prove that arrays were not edited or that these
        sampling and clean-inlier assumptions hold. For external scores use
        ``FPRCertificate.from_scores``.

        The finite-sample one-sided KS band is deterministic. Confidence is
        simultaneous coverage over the calibration draw, conditional on the
        independently fitted scoring map.

        Args:
            confidence: Simultaneous coverage probability in ``(0, 1)``.

        Returns:
            An immutable simultaneous raw-score FPR certificate.

        Raises:
            RuntimeError: If the numerical KS cutoff is invalid.
        """
        from nonconform._internal.fpr_bounds import validate_result_scope
        from nonconform.fdr import FPRCertificate

        calibration_scores = validate_result_scope(self)
        return FPRCertificate.from_scores(
            calibration_scores,
            confidence=confidence,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return one row per test point with its score, p-value, and weight.

        Columns are ``score`` (anomalous-higher), ``p_value`` when p-values
        were computed, and ``weight`` in weighted mode. When the test batch
        was a pandas DataFrame or Series, its index is kept.

        Returns:
            A new DataFrame.

        Raises:
            ValueError: If the snapshot holds no test scores.

        Examples:
            ```python
            import numpy as np
            from sklearn.ensemble import IsolationForest

            from nonconform import ConformalDetector

            rng = np.random.default_rng(0)
            detector = ConformalDetector(IsolationForest(random_state=0)).fit(
                rng.normal(size=(500, 2))
            )
            detector.compute_p_values(rng.normal(size=(5, 2)))
            print(detector.last_result.to_frame())
            ```
        """
        if self.test_scores is None:
            raise ValueError("This result holds no test scores.")
        columns: dict[str, np.ndarray] = {"score": np.asarray(self.test_scores)}
        if self.p_values is not None:
            columns["p_value"] = np.asarray(self.p_values)
        if self.test_weights is not None:
            columns["weight"] = np.asarray(self.test_weights)
        # Snapshots pickled before _index existed lack the slot.
        index = getattr(self, "_index", None)
        return pd.DataFrame(columns, index=index, copy=True)

    def copy(self) -> ConformalResult:
        """Return a copy with arrays and metadata fully duplicated.

        Returns:
            A new ConformalResult with copied arrays and deep-copied metadata.
        """

        def _copy_arr(arr: np.ndarray | None) -> np.ndarray | None:
            return arr.copy() if arr is not None else None

        copied = ConformalResult(
            p_values=_copy_arr(self.p_values),
            test_scores=_copy_arr(self.test_scores),
            calib_scores=_copy_arr(self.calib_scores),
            test_weights=_copy_arr(self.test_weights),
            calib_weights=_copy_arr(self.calib_weights),
            metadata=deepcopy(self.metadata),
        )
        copied._provenance = self._provenance
        copied._index = getattr(self, "_index", None)
        return copied


__all__ = [
    "AnomalyDetector",
    "ConformalResult",
]
