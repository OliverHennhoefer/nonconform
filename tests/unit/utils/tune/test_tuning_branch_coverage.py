from __future__ import annotations

import builtins
from unittest.mock import patch

import numpy as np
import pytest

import nonconform._internal.tuning as tuning


def test_compute_bandwidth_range_caps_extreme_ratio() -> None:
    near_zero = np.linspace(0.0, 1e-6, 9800, dtype=float)
    high_tail = np.linspace(1000.0, 1000.1, 200, dtype=float)
    data = np.concatenate([near_zero, high_tail])

    bw_min, bw_max = tuning.compute_bandwidth_range(data)

    assert bw_min > 0.0
    assert bw_max > 0.0
    assert bw_max / bw_min <= tuning.MAX_BANDWIDTH_RATIO + 1e-12
    assert np.isclose(bw_max, bw_min * tuning.MAX_BANDWIDTH_RATIO)


def test_require_kdepy_fftkde_raises_helpful_error_when_missing() -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args, **kwargs):
        if name == "KDEpy" or name.startswith("KDEpy."):
            raise ImportError("No module named 'KDEpy'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        with pytest.raises(ImportError, match=r"nonconform\[probabilistic\]"):
            tuning._require_kdepy_fftkde()
