"""Integration tests covering strategy + detector interactions."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

from nonconform.detection import ConformalDetector
from nonconform.strategy import (
    CrossValidation,
    Jackknife,
    JackknifeBootstrap,
    Split,
)
from nonconform.utils.func.enums import Aggregation
from pyod.models.iforest import IForest

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
        lambda: CrossValidation(k=3, plus=True),
        lambda n: n,
    ),
    StrategyCase(
        "cross-validation-standard",
        lambda: CrossValidation(k=3, plus=False),
        lambda n: n,
    ),
    StrategyCase(
        "jackknife",
        lambda: Jackknife(plus=True),
        lambda n: n,
    ),
    StrategyCase(
        "jackknife-bootstrap",
        lambda: JackknifeBootstrap(n_bootstraps=4, plus=True),
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
        aggregation=Aggregation.MEDIAN,
        seed=11,
    )


@pytest.mark.parametrize("case", STRATEGY_CASES, ids=lambda case: case.name)
def test_strategies_generate_expected_calibration(simple_dataset, case):
    """Each strategy should report the correct calibration set length."""
    x_train, x_test, _ = simple_dataset(n_train=40, n_test=16, n_features=4)

    detector = _build_detector(case.factory())
    detector.fit(x_train)
    detector.predict(x_test[:6])

    calibration = detector.calibration_set
    assert len(calibration) == case.expected_calib(len(x_train))
    assert np.isfinite(calibration).all()


PlusCase = namedtuple("PlusCase", "name factory expected_models")

PLUS_CASES: tuple[PlusCase, ...] = (
    PlusCase(
        "cross-validation",
        lambda plus: CrossValidation(k=3, plus=plus),
        lambda plus, _: 3 if plus else 1,
    ),
    PlusCase(
        "jackknife",
        lambda plus: Jackknife(plus=plus),
        lambda plus, n: n if plus else 1,
    ),
    PlusCase(
        "jackknife-bootstrap",
        lambda plus: JackknifeBootstrap(n_bootstraps=3, plus=plus),
        lambda plus, _: 3 if plus else 1,
    ),
)


@pytest.mark.parametrize("case", PLUS_CASES, ids=lambda c: c.name)
def test_plus_variants_change_model_count(simple_dataset, case):
    """Plus variants should keep multiple trained detectors for ensembles."""
    x_train, x_test, _ = simple_dataset(n_train=30, n_test=10, n_features=3)

    det_plus = _build_detector(case.factory(True))
    det_plus.fit(x_train)
    det_plus.predict(x_test[:4])

    det_std = _build_detector(case.factory(False))
    det_std.fit(x_train)
    det_std.predict(x_test[:4])

    assert len(det_plus.detector_set) == case.expected_models(True, len(x_train))
    assert len(det_std.detector_set) == case.expected_models(False, len(x_train))


SeedCase = namedtuple("SeedCase", "name factory")

SEED_CASES = (
    SeedCase("split", lambda: Split(n_calib=0.2)),
    SeedCase("cross-validation", lambda: CrossValidation(k=4, plus=True)),
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

    preds_one = det_one.predict(x_test)
    preds_two = det_two.predict(x_test)

    np.testing.assert_allclose(preds_one, preds_two, atol=1e-8)
