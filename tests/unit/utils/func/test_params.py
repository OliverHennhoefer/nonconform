"""Tests for detector parameter configuration with alias support."""

import logging
import sys

import pytest

from nonconform._internal.config import set_params
from nonconform._internal.random_utils import derive_seed

MIN_CONTAMINATION = sys.float_info.min


class ParamsDetector:
    """Detector double exposing only sklearn-style parameter access."""

    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self


@pytest.mark.parametrize(
    ("params", "seed", "expected"),
    [
        (
            {"contamination": 0.1, "n_jobs": None, "random_state": None},
            99,
            {"contamination": MIN_CONTAMINATION, "n_jobs": -1, "random_state": 99},
        ),
        (
            {"contamination": 0.5, "n_jobs": None, "random_state": 999},
            42,
            {"contamination": MIN_CONTAMINATION, "n_jobs": -1, "random_state": 42},
        ),
        (
            {"n_jobs": None, "random_state": None},
            42,
            {"n_jobs": -1, "random_state": 42},
        ),
        (
            {"contamination": 0.1, "random_state": None},
            42,
            {"contamination": MIN_CONTAMINATION, "random_state": 42},
        ),
        (
            {"contamination": 0.1, "n_jobs": None},
            42,
            {"contamination": MIN_CONTAMINATION, "n_jobs": -1},
        ),
        ({"seed": None}, 42, {"seed": 42}),
        ({"random_seed": None}, 99, {"random_seed": 99}),
        ({"n_threads": None}, None, {"n_threads": -1}),
        (
            {"num_workers": None, "random_state": None},
            42,
            {"num_workers": -1, "random_state": 42},
        ),
        (
            {"random_state": None, "seed": None},
            42,
            {"random_state": 42, "seed": None},
        ),
        ({"random_state": None}, 0, {"random_state": 0}),
        ({"random_state": None}, -42, {"random_state": -42}),
        ({"random_state": None}, 2**31 - 1, {"random_state": 2**31 - 1}),
        ({"random_state": 7}, None, {"random_state": 7}),
    ],
    ids=[
        "all-standard-names",
        "existing-values-overwritten",
        "without-contamination",
        "without-n-jobs",
        "without-random-state",
        "seed-alias",
        "random-seed-alias",
        "n-threads-alias",
        "num-workers-alias",
        "random-state-preferred-over-alias",
        "zero-seed",
        "negative-seed",
        "large-seed",
        "no-seed-keeps-random-state",
    ],
)
def test_set_params_configures_available_parameters(params, seed, expected):
    detector = ParamsDetector(**params)

    assert set_params(detector, seed=seed) is detector
    assert detector.params == expected


@pytest.mark.parametrize(("seed", "warns"), [(42, True), (None, False)])
def test_missing_random_state_warns_only_when_seed_is_given(
    caplog, monkeypatch, seed, warns
):
    monkeypatch.setattr(logging.getLogger("nonconform"), "propagate", True)

    with caplog.at_level(logging.DEBUG, logger="nonconform"):
        set_params(ParamsDetector(), seed=seed)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert ("Reproducibility cannot be guaranteed" in caplog.text) is warns
    assert ("random_state" in caplog.text) is warns


class DefaultsDetector(ParamsDetector):
    """Detector double that declares its parallelism default like PyOD."""

    def __init__(self, n_jobs=1, contamination=0.1):
        super().__init__(n_jobs=n_jobs, contamination=contamination)


@pytest.mark.parametrize(
    ("detector", "key", "expected"),
    [
        (ParamsDetector(n_jobs=None), "n_jobs", -1),
        (DefaultsDetector(), "n_jobs", -1),
        (DefaultsDetector(n_jobs=2), "n_jobs", 2),
        (ParamsDetector(n_jobs=3), "n_jobs", 3),
        (ParamsDetector(n_threads=4), "n_threads", 4),
    ],
    ids=[
        "unset-none",
        "declared-default",
        "user-value",
        "unknown-default-kept",
        "alias-user-value",
    ],
)
def test_parallelism_is_filled_only_when_left_at_default(detector, key, expected):
    set_params(detector, seed=None)

    assert detector.params[key] == expected


def test_random_iteration_derives_reproducible_distinct_seeds():
    def configured_seed(seed, iteration, random_iteration=True):
        detector = ParamsDetector(random_state=None)
        set_params(
            detector, seed=seed, random_iteration=random_iteration, iteration=iteration
        )
        return detector.params["random_state"]

    seeds = [configured_seed(42, iteration) for iteration in range(100)]

    assert len(set(seeds[:10])) == 10
    assert len(set(seeds)) >= 99
    assert all(0 <= seed < 2**32 for seed in seeds)
    assert 0 <= configured_seed(42, 1_000_000) < 2**32
    assert configured_seed(42, 7) == configured_seed(42, 7)
    assert configured_seed(42, 0) != configured_seed(99, 0)
    assert configured_seed(42, 10, random_iteration=False) == 42
    assert configured_seed(42, None) == 42


def test_derive_seed_requires_at_least_one_part() -> None:
    with pytest.raises(ValueError, match="At least one seed part"):
        derive_seed()
