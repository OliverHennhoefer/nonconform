"""Strategies print as the constructor call that configured them."""

import pytest

from nonconform import CrossValidation, DerandomizedSplits, JackknifeBootstrap, Split


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (Split(), "Split()"),
        (Split(n_calib=0.2), "Split(n_calib=0.2)"),
        (Split(1_000), "Split(n_calib=1000)"),
        (CrossValidation(), "CrossValidation()"),
        (CrossValidation(k=10, shuffle=False), "CrossValidation(k=10, shuffle=False)"),
        (CrossValidation.jackknife(), "CrossValidation(k=None, shuffle=False)"),
        (JackknifeBootstrap(), "JackknifeBootstrap()"),
        (
            JackknifeBootstrap(50, "median"),
            "JackknifeBootstrap(n_bootstraps=50, aggregation_method='median')",
        ),
        (DerandomizedSplits(), "DerandomizedSplits()"),
        (
            DerandomizedSplits(3, 200, alpha_bh=0.01),
            "DerandomizedSplits(n_repetitions=3, n_calib=200, alpha_bh=0.01)",
        ),
    ],
)
def test_repr_shows_non_default_constructor_arguments(strategy, expected):
    assert repr(strategy) == expected
    assert str(strategy) == expected


def test_repr_shows_single_model_mode(caplog):
    assert repr(CrossValidation(k=3, mode="single_model")) == (
        "CrossValidation(k=3, mode='single_model')"
    )
    assert repr(JackknifeBootstrap(mode="single_model")) == (
        "JackknifeBootstrap(mode='single_model')"
    )


def test_custom_strategy_without_parameters_prints_its_name():
    from nonconform.resampling import BaseStrategy

    class Holdout(BaseStrategy):
        def fit_calibrate(self, x, detector, seed=None, weighted=False):
            raise NotImplementedError

        @property
        def calibration_ids(self):
            return None

    strategy = Holdout()

    assert repr(strategy) == "Holdout()"
    assert strategy.calibration_ids is None
    with pytest.raises(NotImplementedError):
        strategy.fit_calibrate(None, None)
