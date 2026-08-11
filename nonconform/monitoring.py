"""Sequential conformal monitoring with exact randomized ranks.

This module supplies the validity-critical first half of an exchangeability
martingale workflow: a frozen anomaly scoring rule followed by randomized
sequential ranks.  The resulting p-values can be consumed by the betting
martingales in :mod:`nonconform.martingales`.

The existing :class:`nonconform.ConformalDetector` remains the batch and
pointwise conformal API.  Its fixed-calibration p-values are deliberately not
modified by this module.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort_right
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from nonconform._internal import set_params
from nonconform.adapters import (
    adapt,
    apply_score_polarity,
    resolve_implicit_score_polarity,
    resolve_score_polarity,
)
from nonconform.detector import ConformalDetector
from nonconform.enums import ScorePolarity
from nonconform.martingales import (
    BaseMartingale,
    MartingaleState,
    SimpleJumperMartingale,
)
from nonconform.resampling import Split
from nonconform.structures import AnomalyDetector

Tail = Literal["upper", "lower"]
ScorePolarityInput = (
    ScorePolarity | Literal["auto", "higher_is_anomalous", "higher_is_normal"] | None
)


def _validate_seed(seed: int | None) -> int | None:
    """Validate a reproducibility seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer or None.")
    return seed


def _validate_score(score: float) -> float:
    """Return a finite scalar score."""
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("score must be a numeric scalar.") from exc
    if not np.isfinite(value):
        raise ValueError("score must be finite.")
    return value


def _as_2d_array(name: str, values: Any) -> np.ndarray:
    """Normalize a feature batch into a finite two-dimensional array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric feature matrix.") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


class SequentialRankConformalizer:
    """Generate exact randomized sequential conformal p-values.

    At each update, the new score is ranked among all scores observed by this
    object, including the new score itself.  Independent randomized tie
    breaking makes the resulting p-values IID ``Uniform(0, 1)`` when the score
    sequence is exchangeable.

    Args:
        tail: ``"upper"`` when larger scores are more extreme and ``"lower"``
            when smaller scores are more extreme. Defaults to ``"upper"``.
        seed: Optional seed for a persistent random number generator.

    Notes:
        History grows without a sliding window.  The current implementation
        uses an exact sorted list: rank queries are logarithmic, while insertion
        is linear in the history length.
    """

    def __init__(self, *, tail: Tail = "upper", seed: int | None = None) -> None:
        if tail not in {"upper", "lower"}:
            raise ValueError("tail must be either 'upper' or 'lower'.")
        self.tail: Tail = tail
        self.seed = _validate_seed(seed)
        self.reset()

    @property
    def count(self) -> int:
        """Number of scores currently in the sequential rank history."""
        return len(self._sorted_scores)

    @property
    def scores(self) -> np.ndarray:
        """Return the sorted score history as a copy."""
        return np.asarray(self._sorted_scores, dtype=float).copy()

    def reset(self) -> None:
        """Clear score history and restore the initial RNG state."""
        self._sorted_scores: list[float] = []
        self._rng = np.random.default_rng(self.seed)

    def prime(self, score: float) -> Self:
        """Add one score to rank history without producing a p-value."""
        insort_right(self._sorted_scores, _validate_score(score))
        return self

    def prime_many(self, scores: Any) -> Self:
        """Add scores to rank history without consuming RNG draws."""
        array = np.asarray(scores)
        if array.ndim != 1:
            raise ValueError("scores must be a one-dimensional sequence.")
        for score in array:
            self.prime(float(score))
        return self

    def update(self, score: float) -> float:
        """Insert one score and return its randomized sequential p-value."""
        value = _validate_score(score)
        n_previous = len(self._sorted_scores)
        left = bisect_left(self._sorted_scores, value)
        right = bisect_right(self._sorted_scores, value)
        n_equal_including_new = right - left + 1
        if self.tail == "upper":
            n_more_extreme = n_previous - right
        else:
            n_more_extreme = left

        random_tie_fraction = float(self._rng.random())
        p_value = (n_more_extreme + random_tie_fraction * n_equal_including_new) / (
            n_previous + 1
        )
        insort_right(self._sorted_scores, value)
        return float(p_value)

    def update_many(self, scores: Any) -> np.ndarray:
        """Process a one-dimensional score sequence in order."""
        array = np.asarray(scores)
        if array.ndim != 1:
            raise ValueError("scores must be a one-dimensional sequence.")
        return np.asarray([self.update(float(score)) for score in array], dtype=float)


@dataclass(slots=True, frozen=True)
class MonitorState:
    """Snapshot from one rigorous sequential monitoring update."""

    rank_step: int
    score: float
    martingale_state: MartingaleState

    @property
    def evidence_step(self) -> int:
        """Number of observations included in the evidence process."""
        return self.martingale_state.step

    @property
    def p_value(self) -> float:
        """Randomized sequential conformal p-value."""
        return self.martingale_state.p_value

    @property
    def e_value(self) -> float:
        """Stepwise betting factor used for this update."""
        return self.martingale_state.e_value

    @property
    def log_e_value(self) -> float:
        """Natural logarithm of the stepwise betting factor."""
        return self.martingale_state.log_e_value

    @property
    def martingale(self) -> float:
        """Cumulative product-martingale value."""
        return self.martingale_state.martingale

    @property
    def restarted_martingale(self) -> float:
        """Harmonic restart-mixture e-process value."""
        return self.martingale_state.restarted_martingale

    @property
    def triggered_alarms(self) -> tuple[str, ...]:
        """Alarm names whose configured thresholds are currently crossed."""
        return self.martingale_state.triggered_alarms


class ExchangeabilityMonitor:
    """Monitor a stream using a frozen scorer and sequential conformal ranks.

    The scorer is fitted once and then applied coordinate-wise to reference and
    stream observations.  ``prime`` establishes rank history without starting
    evidence; ``update`` generates a randomized sequential p-value and sends it
    to the configured martingale.

    Args:
        detector: PyOD, scikit-learn, or custom anomaly detector.
        conformalizer: Stateful sequential rank conformalizer. Defaults to an
            upper-tail :class:`SequentialRankConformalizer`.
        martingale: P-value betting martingale. Defaults to Simple Jumper.
        score_polarity: Detector score direction, following
            :class:`nonconform.ConformalDetector` semantics.
        seed: Seed propagated to the scorer and default conformalizer.

    Notes:
        Exact validity requires the priming and monitored scores to be
        exchangeable under the null, conditional on a fixed training-only
        scoring construction.  Do not refit the scorer during an episode.
    """

    def __init__(
        self,
        detector: Any,
        *,
        conformalizer: SequentialRankConformalizer | None = None,
        martingale: BaseMartingale | None = None,
        score_polarity: ScorePolarityInput = None,
        seed: int | None = None,
    ) -> None:
        self.seed = _validate_seed(seed)
        adapted_detector = adapt(detector)
        if score_polarity is None:
            resolved_polarity = resolve_implicit_score_polarity(adapted_detector)
        else:
            resolved_polarity = resolve_score_polarity(adapted_detector, score_polarity)
        normalized_detector = apply_score_polarity(adapted_detector, resolved_polarity)
        self.detector: AnomalyDetector = set_params(
            deepcopy(normalized_detector), self.seed
        )
        self.conformalizer = (
            conformalizer
            if conformalizer is not None
            else SequentialRankConformalizer(seed=self.seed)
        )
        self.martingale = (
            martingale if martingale is not None else SimpleJumperMartingale()
        )
        if self.conformalizer.count != 0:
            raise ValueError("conformalizer must have empty history at construction.")
        if self.martingale.state.step != 0:
            raise ValueError("martingale must be reset at construction.")
        self._is_fitted = False
        self._n_features_in: int | None = None

    @classmethod
    def from_split_detector(
        cls,
        detector: ConformalDetector,
        *,
        conformalizer: SequentialRankConformalizer | None = None,
        martingale: BaseMartingale | None = None,
        seed: int | None = None,
    ) -> ExchangeabilityMonitor:
        """Create a rigorous monitor from a fitted unweighted Split detector.

        The fitted scoring model is copied and frozen.  Existing calibration
        scores initialize sequential rank history; they are not reused as a
        fixed empirical CDF and do not contribute to martingale capital.
        """
        if not isinstance(detector, ConformalDetector):
            raise TypeError("detector must be a ConformalDetector instance.")
        if not detector.is_fitted:
            raise NotFittedError("ConformalDetector must be fitted first.")
        if not isinstance(detector.strategy, Split):
            raise ValueError("from_split_detector requires a Split strategy.")
        if detector._is_weighted_mode:
            raise ValueError("from_split_detector does not support weighted mode.")
        fitted_models = detector.detector_set
        if len(fitted_models) != 1:
            raise ValueError("from_split_detector requires exactly one fitted model.")

        resolved_conformalizer = (
            conformalizer
            if conformalizer is not None
            else SequentialRankConformalizer(seed=seed)
        )
        monitor = cls(
            fitted_models[0],
            conformalizer=resolved_conformalizer,
            martingale=martingale,
            score_polarity="higher_is_anomalous",
            seed=None,
        )
        # Preserve the fitted scorer exactly. The regular constructor configures
        # unfitted estimators for training, which must not be repeated here.
        monitor.detector = deepcopy(fitted_models[0])
        monitor.seed = _validate_seed(seed)
        monitor._is_fitted = True
        monitor._n_features_in = detector._n_features_in
        monitor.conformalizer.prime_many(detector.calibration_set)
        return monitor

    @property
    def is_fitted(self) -> bool:
        """Whether the scoring rule is fitted and frozen."""
        return self._is_fitted

    @property
    def state(self) -> MonitorState | None:
        """Return the latest monitoring state, or None before the first update."""
        return getattr(self, "_last_state", None)

    def fit(self, x: pd.DataFrame | np.ndarray) -> Self:
        """Fit the scoring rule once on a proper training set."""
        x_array = _as_2d_array("x", x)
        self.detector.fit(x_array)
        self._is_fitted = True
        self._n_features_in = int(x_array.shape[1])
        self.reset()
        return self

    def reset(self) -> None:
        """Reset rank and evidence state while retaining the fitted scorer.

        A reset starts a new monitoring episode.  Repeated episodes need their
        own error-budget accounting if a lifetime false-alarm guarantee is
        required.
        """
        self.conformalizer.reset()
        self.martingale.reset()
        self._last_state: MonitorState | None = None

    def prime(self, x: pd.DataFrame | np.ndarray) -> Self:
        """Score reference observations and add them only to rank history."""
        self._require_fitted()
        if self.martingale.state.step != 0:
            raise RuntimeError(
                "prime() is unavailable after evidence monitoring starts."
            )
        x_array = self._validate_feature_batch("x", x)
        for row in x_array:
            self.conformalizer.prime(self._score_one(row))
        return self

    def update(self, x: pd.Series | np.ndarray) -> MonitorState:
        """Score one observation and update sequential evidence."""
        self._require_fitted()
        try:
            x_array = np.asarray(x, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("x must be a finite numeric feature vector.") from exc
        if x_array.ndim != 1:
            raise ValueError("x must be a one-dimensional feature vector.")
        self._validate_feature_count(x_array)
        score = self._score_one(x_array)
        p_value = self.conformalizer.update(score)
        martingale_state = self.martingale.update(p_value)
        state = MonitorState(
            rank_step=self.conformalizer.count,
            score=score,
            martingale_state=martingale_state,
        )
        self._last_state = state
        return state

    def update_many(self, x: pd.DataFrame | np.ndarray) -> list[MonitorState]:
        """Update sequentially for every row in a feature batch."""
        self._require_fitted()
        x_array = self._validate_feature_batch("x", x)
        return [self.update(row) for row in x_array]

    def _require_fitted(self) -> None:
        """Raise when the scoring model is unavailable."""
        if not self._is_fitted:
            raise NotFittedError("ExchangeabilityMonitor must be fitted first.")

    def _validate_feature_count(self, x: np.ndarray) -> None:
        """Validate a feature vector against fitted dimensionality."""
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite.")
        if self._n_features_in is not None and x.shape[0] != self._n_features_in:
            raise ValueError(
                f"x has {x.shape[0]} features, but the fitted scorer expects "
                f"{self._n_features_in}."
            )

    def _validate_feature_batch(self, name: str, x: Any) -> np.ndarray:
        """Validate a feature batch against fitted dimensionality."""
        x_array = _as_2d_array(name, x)
        if self._n_features_in is not None and x_array.shape[1] != self._n_features_in:
            raise ValueError(
                f"{name} has {x_array.shape[1]} features, but the fitted scorer "
                f"expects {self._n_features_in}."
            )
        return x_array

    def _score_one(self, x: np.ndarray) -> float:
        """Apply the frozen scoring rule to exactly one observation."""
        scores = np.asarray(
            self.detector.decision_function(np.asarray(x)[np.newaxis, :]),
            dtype=float,
        ).ravel()
        if scores.shape != (1,):
            raise ValueError("detector must return exactly one score per observation.")
        return _validate_score(float(scores[0]))


__all__ = [
    "ExchangeabilityMonitor",
    "MonitorState",
    "SequentialRankConformalizer",
]
