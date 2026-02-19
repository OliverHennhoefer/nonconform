import numpy as np
import pytest

import nonconform._internal.math_utils as math_utils
from nonconform._internal.random_utils import derive_seed


def test_derive_seed_requires_at_least_one_part() -> None:
    with pytest.raises(ValueError, match="At least one seed part"):
        derive_seed()


def test_normalize_bootstrap_aggregation_method_rejects_minimum() -> None:
    with pytest.raises(ValueError, match="Unsupported bootstrap aggregation method"):
        math_utils.normalize_bootstrap_aggregation_method("minimum")


def test_aggregate_assert_never_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    def _invalid_normalizer(method: str) -> str:
        return "invalid"  # pragma: no cover - runtime-only guard injection

    monkeypatch.setattr(math_utils, "normalize_aggregation_method", _invalid_normalizer)
    with pytest.raises(AssertionError):
        math_utils.aggregate("mean", np.array([[1.0, 2.0]]))
