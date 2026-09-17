"""Integration tests for raw-score false-positive-rate certificates."""

from __future__ import annotations

from typing import Self

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from nonconform import (
    ConformalDetector,
    CrossValidation,
    Split,
    logistic_weight_estimator,
)
from nonconform.scoring import ConditionalEmpirical


class DistanceDetector:
    """Small detector with an explicit score-direction variant."""

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse
        self.center_: np.ndarray | None = None

    def fit(self, X, y=None) -> Self:
        _ = y
        self.center_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X):
        if self.center_ is None:
            raise RuntimeError("detector is not fitted")
        scores = np.linalg.norm(X - self.center_, axis=1)
        return -scores if self.reverse else scores

    def get_params(self, deep=True):
        _ = deep
        return {"reverse": self.reverse}

    def set_params(self, **params) -> Self:
        if "reverse" in params:
            self.reverse = params["reverse"]
        return self


def _data(seed: int = 42):
    rng = np.random.default_rng(seed)
    x_reference = rng.normal(size=(120, 3))
    x_test = rng.normal(size=(25, 3))
    return x_reference, x_test


def test_integrated_detector_and_result_certificates_match():
    x_reference, x_test = _data()
    detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=Split(n_calib=0.25),
        seed=7,
    ).fit(x_reference)

    detector_certificate = detector.fpr_bounds(
        confidence=0.8,
        n_resamples=30,
        seed=11,
    )
    scores = detector.score_samples(x_test)
    result = detector.last_result
    assert result is not None
    result_certificate = result.fpr_bounds(
        confidence=0.8,
        n_resamples=30,
        seed=11,
    )

    np.testing.assert_allclose(
        detector_certificate.to_frame(), result_certificate.to_frame()
    )
    mask = detector_certificate.select(scores, target_fpr=0.2)
    assert mask.dtype == bool
    assert mask.shape == (len(x_test),)


def test_detector_fpr_bounds_does_not_replace_last_result():
    x_reference, x_test = _data()
    detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=Split(n_calib=0.25),
        seed=7,
    ).fit(x_reference)
    detector.compute_p_values(x_test)
    before = detector.last_result
    assert before is not None and before.p_values is not None

    certificate = detector.fpr_bounds(n_resamples=20, seed=11)
    after = detector.last_result

    assert certificate.method == "mc_ks"
    assert after is not None and after.p_values is not None
    np.testing.assert_array_equal(after.p_values, before.p_values)


def test_detached_calibration_produces_native_certificate():
    x_fit, x_calibration = _data(1)[0], _data(2)[0]
    base_detector = DistanceDetector().fit(x_fit)
    detector = ConformalDetector(
        detector=base_detector,
        strategy=Split(n_calib=0.25),
        seed=9,
    ).calibrate(x_calibration)

    certificate = detector.fpr_bounds(n_resamples=20, seed=4)

    assert certificate.n_calibration == len(x_calibration)
    assert certificate.calibration_scores.shape == (len(x_calibration),)


def test_native_score_polarity_is_normalized_before_certification():
    x_reference, _ = _data()
    normality_detector = ConformalDetector(
        detector=DistanceDetector(reverse=True),
        strategy=Split(n_calib=0.25),
        score_polarity="higher_is_normal",
        seed=7,
    ).fit(x_reference)
    anomaly_detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=Split(n_calib=0.25),
        seed=7,
    ).fit(x_reference)

    np.testing.assert_allclose(
        normality_detector.fpr_bounds(n_resamples=20, seed=4).calibration_scores,
        anomaly_detector.fpr_bounds(n_resamples=20, seed=4).calibration_scores,
    )


def test_unfitted_detector_rejects_fpr_certificate():
    detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=Split(n_calib=0.25),
    )

    with pytest.raises(NotFittedError):
        detector.fpr_bounds()


@pytest.mark.parametrize(
    "strategy",
    [CrossValidation(k=3)],
)
def test_unsupported_native_strategy_is_rejected(strategy):
    x_reference, _ = _data()
    detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=strategy,
        seed=7,
    ).fit(x_reference)

    with pytest.raises(ValueError, match="split or detached"):
        detector.fpr_bounds()


def test_conditional_estimation_is_rejected():
    x_reference, _ = _data()
    detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=Split(n_calib=0.25),
        estimation=ConditionalEmpirical(method="dkwm"),
        seed=7,
    ).fit(x_reference)

    with pytest.raises(ValueError, match="empirical conformal calibration"):
        detector.fpr_bounds()


def test_weighted_calibration_is_rejected():
    x_reference, _ = _data()
    detector = ConformalDetector(
        detector=DistanceDetector(),
        strategy=Split(n_calib=0.25),
        weight_estimator=logistic_weight_estimator(),
        seed=7,
    ).fit(x_reference)

    with pytest.raises(ValueError, match="unweighted"):
        detector.fpr_bounds()
