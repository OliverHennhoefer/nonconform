import builtins
import sys

import pytest

from nonconform._internal.tuning import tune_kde_hyperparameters
from nonconform.enums import Kernel
from nonconform.scoring import Probabilistic


def _block_import(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):
        if name == package or name.startswith(f"{package}."):
            raise ImportError(f"No module named '{package}'")
        return real_import(name, *args, **kwargs)

    # Remove cached modules so guarded import is exercised reliably.
    to_remove = [
        key for key in sys.modules if key == package or key.startswith(f"{package}.")
    ]
    for key in to_remove:
        monkeypatch.delitem(sys.modules, key, raising=False)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_internal_namespace_does_not_export_tuning_symbol() -> None:
    with pytest.raises(ImportError, match="tune_kde_hyperparameters"):
        exec("from nonconform._internal import tune_kde_hyperparameters", {})


def test_heuristic_mode_does_not_require_optuna(
    monkeypatch: pytest.MonkeyPatch,
    sample_calibration_data,
) -> None:
    _block_import(monkeypatch, "optuna")
    data = sample_calibration_data(n_samples=60)
    result = tune_kde_hyperparameters(data, [Kernel.GAUSSIAN], n_trials=0)
    assert result["study"] is None


def test_tuning_raises_helpful_error_without_optuna(
    monkeypatch: pytest.MonkeyPatch,
    sample_calibration_data,
) -> None:
    _block_import(monkeypatch, "optuna")
    data = sample_calibration_data(n_samples=60)
    with pytest.raises(ImportError, match=r"nonconform\[probabilistic\]"):
        tune_kde_hyperparameters(data, [Kernel.GAUSSIAN], n_trials=1, cv_folds=2)


def test_probabilistic_raises_helpful_error_without_kdepy(
    monkeypatch: pytest.MonkeyPatch,
    sample_calibration_data,
) -> None:
    _block_import(monkeypatch, "KDEpy")
    estimator = Probabilistic(n_trials=0)
    data = sample_calibration_data(n_samples=60)
    with pytest.raises(ImportError, match=r"nonconform\[probabilistic\]"):
        estimator._fit_kde(data, None)
