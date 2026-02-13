"""Integration tests covering strategy + detector interactions."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

pytest.importorskip("pyod", reason="pyod not installed")
from pyod.models.iforest import IForest

from nonconform import (
    ConformalDetector,
    CrossValidation,
    JackknifeBootstrap,
    Split,
)

StrategyCase = namedtuple("StrategyCase", "name factory expected_calib")

STRATEGY_CASES = [
    StrategyCase(
        "split-fraction",
        lambda: Split(n_calib=0.25),
        lambda n: int(n * 0.25),
    ),
    StrategyCase(
        "split-absolute",
        lambda: Split(n_calib=8),
        lambda n: 8,
    ),
    StrategyCase(
        "cross-validation-plus",
        lambda: CrossValidation(k=3, mode="plus"),
        lambda n: n,
    ),
    StrategyCase(
        "cross-validation-standard",
        lambda: CrossValidation(k=3, mode="single_model"),
        lambda n: n,
    ),
    StrategyCase(
        "jackknife",
        lambda: CrossValidation.jackknife(mode="plus"),
        lambda n: n,
    ),
    StrategyCase(
        "jackknife-bootstrap",
        lambda: JackknifeBootstrap(n_bootstraps=4, mode="plus"),
        lambda n: n,
    ),
]


def _build_detector(strategy):
    return ConformalDetector(
        detector=IForest(
            n_estimators=15,
            max_samples=0.8,
            random_state=0,
        ),
        strategy=strategy,
        aggregation="median",
        seed=11,
    )


@pytest.mark.parametrize("case", STRATEGY_CASES, ids=lambda case: case.name)
def test_strategies_generate_expected_calibration(simple_dataset, case):
    """Each strategy should report the correct calibration set length."""
    x_train, x_test, _ = simple_dataset(n_train=40, n_test=16, n_features=4)

    detector = _build_detector(case.factory())
    detector.fit(x_train)
    detector.compute_p_values(x_test[:6])

    calibration = detector.calibration_set
    assert len(calibration) == case.expected_calib(len(x_train))
    assert np.isfinite(calibration).all()


ModeCase = namedtuple("ModeCase", "name factory expected_models")

MODE_CASES: tuple[ModeCase, ...] = (
    ModeCase(
        "cross-validation",
        lambda mode: CrossValidation(k=3, mode=mode),
        lambda mode, _: 3 if mode == "plus" else 1,
    ),
    ModeCase(
        "jackknife",
        lambda mode: CrossValidation.jackknife(mode=mode),
        lambda mode, n: n if mode == "plus" else 1,
    ),
    ModeCase(
        "jackknife-bootstrap",
        lambda mode: JackknifeBootstrap(n_bootstraps=3, mode=mode),
        lambda mode, _: 3 if mode == "plus" else 1,
    ),
)


@pytest.mark.parametrize("case", MODE_CASES, ids=lambda c: c.name)
def test_mode_variants_change_model_count(simple_dataset, case):
    """Mode selection should control how many trained detectors are retained."""
    x_train, x_test, _ = simple_dataset(n_train=30, n_test=10, n_features=3)

    det_plus = _build_detector(case.factory("plus"))
    det_plus.fit(x_train)
    det_plus.compute_p_values(x_test[:4])

    det_std = _build_detector(case.factory("single_model"))
    det_std.fit(x_train)
    det_std.compute_p_values(x_test[:4])

    assert len(det_plus.detector_set) == case.expected_models("plus", len(x_train))
    assert len(det_std.detector_set) == case.expected_models(
        "single_model", len(x_train)
    )


SeedCase = namedtuple("SeedCase", "name factory")

SEED_CASES = (
    SeedCase("split", lambda: Split(n_calib=0.2)),
    SeedCase("cross-validation", lambda: CrossValidation(k=4, mode="plus")),
    SeedCase("jackknife-bootstrap", lambda: JackknifeBootstrap(n_bootstraps=3)),
)


@pytest.mark.parametrize("case", SEED_CASES, ids=lambda c: c.name)
def test_strategy_runs_are_reproducible(simple_dataset, case):
    """Same seed should give identical p-values per strategy."""
    x_train, x_test, _ = simple_dataset(n_train=32, n_test=12, n_features=3)

    det_one = _build_detector(case.factory())
    det_two = _build_detector(case.factory())

    det_one.fit(x_train)
    det_two.fit(x_train)

    preds_one = det_one.compute_p_values(x_test)
    preds_two = det_two.compute_p_values(x_test)

    np.testing.assert_allclose(preds_one, preds_two, atol=1e-8)
