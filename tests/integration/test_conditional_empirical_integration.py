"""Integration tests for ConditionalEmpirical estimation."""

from __future__ import annotations

import numpy as np
import pytest
from pyod.models.iforest import IForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.scoring import ConditionalEmpirical


def _build_conditional_detector(
    *,
    estimation: ConditionalEmpirical,
    weighted: bool = False,
) -> ConformalDetector:
    return ConformalDetector(
        detector=IForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=estimation,
        weight_estimator=logistic_weight_estimator() if weighted else None,
        seed=41,
    )


def test_conditional_empirical_outputs_valid_p_values(simple_dataset) -> None:
    x_train, x_test, _ = simple_dataset(n_train=90, n_test=28, n_features=4)
    detector = _build_conditional_detector(
        estimation=ConditionalEmpirical(method="simes", delta=0.1),
    )

    detector.fit(x_train)
    p_values = detector.compute_p_values(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))
    assert detector.last_result is not None
    assert detector.last_result.p_values is not None


def test_conditional_empirical_reproducible_with_seed(simple_dataset) -> None:
    x_train, x_test, _ = simple_dataset(n_train=84, n_test=24, n_features=3)
    kwargs = dict(
        method="simes",
        tie_break="randomized",
        delta=0.1,
    )

    first = _build_conditional_detector(estimation=ConditionalEmpirical(**kwargs))
    second = _build_conditional_detector(estimation=ConditionalEmpirical(**kwargs))

    first.fit(x_train)
    second.fit(x_train)

    p_first = first.compute_p_values(x_test)
    p_second = second.compute_p_values(x_test)
    np.testing.assert_array_equal(p_first, p_second)


def test_conditional_empirical_rejects_weighted_mode(simple_dataset) -> None:
    x_train, x_test, _ = simple_dataset(n_train=90, n_test=30, n_features=5)
    detector = _build_conditional_detector(
        estimation=ConditionalEmpirical(method="dkwm", delta=0.1),
        weighted=True,
    )
    detector.fit(x_train)

    with pytest.raises(ValueError, match="does not support weighted p-values"):
        detector.compute_p_values(x_test)


def test_conditional_empirical_all_methods_run_through_detector_without_fallback(
    simple_dataset,
) -> None:
    x_train, x_test, _ = simple_dataset(n_train=120, n_test=32, n_features=4, seed=41)
    p_values_by_method: dict[str, np.ndarray] = {}

    for method in ["mc", "simes", "dkwm", "asymptotic"]:
        kwargs: dict[str, object] = {"method": method, "delta": 0.1}
        if method == "mc":
            kwargs["mc_num_simulations"] = 200

        detector = _build_conditional_detector(
            estimation=ConditionalEmpirical(**kwargs),
        )
        detector.fit(x_train)

        p_values = detector.compute_p_values(x_test)
        selections = detector.select(x_test, alpha=0.2)

        assert p_values.shape == (len(x_test),)
        assert np.all(np.isfinite(p_values))
        assert np.all((0.0 <= p_values) & (p_values <= 1.0))
        assert selections.dtype == bool
        assert selections.shape == (len(x_test),)
        assert detector.last_result is not None
        assert detector.last_result.p_values is not None

        p_values_by_method[method] = p_values

    assert not np.allclose(
        p_values_by_method["mc"],
        p_values_by_method["dkwm"],
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        p_values_by_method["asymptotic"],
        p_values_by_method["dkwm"],
        rtol=0.0,
        atol=1e-12,
    )
