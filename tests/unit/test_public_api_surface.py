from enum import Enum

import pytest

import nonconform

CURATED_ROOT_EXPORTS = [
    "ConformalDetector",
    "Split",
    "CrossValidation",
    "JackknifeBootstrap",
    "Empirical",
    "Probabilistic",
    "logistic_weight_estimator",
    "forest_weight_estimator",
]

REMOVED_ROOT_SYMBOLS = [
    "PYOD_AVAILABLE",
    "Aggregation",
    "AnomalyDetector",
    "BaseConformalDetector",
    "BaseEstimation",
    "BaseStrategy",
    "BaseWeightEstimator",
    "BootstrapBaggedWeightEstimator",
    "ConformalResult",
    "Distribution",
    "IdentityWeightEstimator",
    "Kernel",
    "Pruning",
    "PyODAdapter",
    "SklearnWeightEstimator",
    "adapt",
    "aggregate",
    "false_discovery_rate",
    "statistical_power",
    "weighted_bh",
    "weighted_false_discovery_control",
]


def test_root_all_is_exact_curated_surface():
    assert nonconform.__all__ == CURATED_ROOT_EXPORTS


def test_star_import_exposes_only_curated_symbols():
    namespace: dict[str, object] = {}
    exec("from nonconform import *", namespace)
    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == set(CURATED_ROOT_EXPORTS)


def test_removed_root_symbols_are_not_attributes():
    for symbol in REMOVED_ROOT_SYMBOLS:
        assert not hasattr(nonconform, symbol)


@pytest.mark.parametrize("symbol", REMOVED_ROOT_SYMBOLS)
def test_removed_root_import_raises_import_error(symbol: str):
    with pytest.raises(ImportError, match=symbol):
        exec(f"from nonconform import {symbol}", {})


def test_enums_module_exports_expected_symbols():
    from nonconform.enums import (
        Distribution,
        Kernel,
        Pruning,
        ScorePolarity,
    )

    assert issubclass(Distribution, Enum)
    assert issubclass(Kernel, Enum)
    assert issubclass(Pruning, Enum)
    assert issubclass(ScorePolarity, Enum)


def test_aggregation_removed_from_enums_module():
    with pytest.raises(ImportError, match="Aggregation"):
        exec("from nonconform.enums import Aggregation", {})


def test_metrics_module_exports_expected_symbols():
    from nonconform.metrics import aggregate, false_discovery_rate, statistical_power

    assert callable(aggregate)
    assert callable(false_discovery_rate)
    assert callable(statistical_power)
