"""Unit tests for detection/adapter.py."""

from copy import copy, deepcopy
from typing import Any, Self
from unittest.mock import MagicMock

import numpy as np
import pytest

from nonconform.detection.adapter import PyODAdapter, _looks_like_pyod, adapt
from nonconform.detection.protocol import AnomalyDetector


class MockProtocolCompliantDetector:
    """Mock detector that implements AnomalyDetector protocol."""

    def __init__(self):
        self.fitted = False
        self._params = {"param1": 1, "param2": "value"}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self.fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.standard_normal(len(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        return self


class MockNonCompliantDetector:
    """Mock detector missing required methods."""

    def fit(self, X):
        pass

    # Missing: decision_function, get_params, set_params


class MockPartialDetector:
    """Mock detector missing some required methods."""

    def fit(self, X):
        pass

    def decision_function(self, X):
        return np.array([1.0])

    # Missing: get_params, set_params


class TestAdaptFunction:
    """Tests for adapt() function."""

    def test_adapt_protocol_compliant_passthrough(self):
        """Protocol-compliant detectors pass through unchanged."""
        detector = MockProtocolCompliantDetector()
        adapted = adapt(detector)
        assert adapted is detector

    def test_adapt_non_compliant_raises_type_error(self):
        """Non-compliant detectors raise TypeError with clear message."""
        detector = MockNonCompliantDetector()
        with pytest.raises(TypeError) as exc_info:
            adapt(detector)
        assert "Missing methods" in str(exc_info.value)
        assert "decision_function" in str(exc_info.value)
        assert "get_params" in str(exc_info.value)
        assert "set_params" in str(exc_info.value)

    def test_adapt_partial_detector_lists_missing_methods(self):
        """Partial detectors get clear error about missing methods."""
        detector = MockPartialDetector()
        with pytest.raises(TypeError) as exc_info:
            adapt(detector)
        assert "get_params" in str(exc_info.value)
        assert "set_params" in str(exc_info.value)
        # Should NOT list methods that exist
        assert "fit" not in str(exc_info.value)
        assert "decision_function" not in str(exc_info.value)


class TestPyODAdapter:
    """Tests for PyODAdapter class."""

    @pytest.fixture
    def mock_pyod_detector(self):
        """Create mock that looks like a PyOD detector."""
        detector = MagicMock()
        detector.fit.return_value = detector
        detector.decision_function.return_value = np.array([0.1, 0.2, 0.3])
        detector.get_params.return_value = {"n_estimators": 100}
        detector.set_params.return_value = detector
        return detector

    def test_adapter_delegates_fit(self, mock_pyod_detector):
        """Adapter delegates fit() to wrapped detector."""
        adapter = PyODAdapter(mock_pyod_detector)
        X = np.array([[1, 2], [3, 4]])
        result = adapter.fit(X)
        mock_pyod_detector.fit.assert_called_once_with(X, None)
        assert result is adapter

    def test_adapter_delegates_decision_function(self, mock_pyod_detector):
        """Adapter delegates decision_function() to wrapped detector."""
        adapter = PyODAdapter(mock_pyod_detector)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scores = adapter.decision_function(X)
        mock_pyod_detector.decision_function.assert_called_once_with(X)
        np.testing.assert_array_equal(scores, np.array([0.1, 0.2, 0.3]))

    def test_adapter_delegates_get_params(self, mock_pyod_detector):
        """Adapter delegates get_params() to wrapped detector."""
        adapter = PyODAdapter(mock_pyod_detector)
        params = adapter.get_params()
        mock_pyod_detector.get_params.assert_called_once_with(deep=True)
        assert params == {"n_estimators": 100}

    def test_adapter_delegates_set_params(self, mock_pyod_detector):
        """Adapter delegates set_params() to wrapped detector."""
        adapter = PyODAdapter(mock_pyod_detector)
        result = adapter.set_params(n_estimators=200)
        mock_pyod_detector.set_params.assert_called_once_with(n_estimators=200)
        assert result is adapter

    def test_adapter_getattr_passthrough(self, mock_pyod_detector):
        """Adapter passes through unknown attributes to wrapped detector."""
        mock_pyod_detector.custom_attr = "custom_value"
        adapter = PyODAdapter(mock_pyod_detector)
        assert adapter.custom_attr == "custom_value"

    def test_adapter_repr(self, mock_pyod_detector):
        """Adapter repr includes wrapped detector."""
        adapter = PyODAdapter(mock_pyod_detector)
        repr_str = repr(adapter)
        assert "PyODAdapter" in repr_str


class TestPyODAdapterCopy:
    """Tests for PyODAdapter copy support."""

    @pytest.fixture
    def copyable_mock_detector(self):
        """Create a copyable mock detector."""
        detector = MockProtocolCompliantDetector()
        return detector

    def test_shallow_copy(self, copyable_mock_detector):
        """Shallow copy creates new adapter with copied detector."""

        class CopyableMock:
            def __init__(self):
                self.value = 42

            def fit(self, X, y=None):
                return self

            def decision_function(self, X):
                return np.array([1.0])

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

            def __copy__(self):
                new = CopyableMock()
                new.value = self.value
                return new

        detector = CopyableMock()
        adapter = PyODAdapter(detector)
        adapter_copy = copy(adapter)

        assert adapter_copy is not adapter
        assert adapter_copy._detector is not adapter._detector
        assert isinstance(adapter_copy, PyODAdapter)

    def test_deep_copy(self):
        """Deep copy creates fully independent adapter."""

        class DeepCopyableMock:
            def __init__(self):
                self.nested = {"key": [1, 2, 3]}

            def fit(self, X, y=None):
                return self

            def decision_function(self, X):
                return np.array([1.0])

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        detector = DeepCopyableMock()
        adapter = PyODAdapter(detector)
        adapter_copy = deepcopy(adapter)

        assert adapter_copy is not adapter
        assert adapter_copy._detector is not adapter._detector
        # Verify deep copy - modifying original doesn't affect copy
        detector.nested["key"].append(4)
        assert 4 not in adapter_copy._detector.nested["key"]


class TestLooksLikePyOD:
    """Tests for _looks_like_pyod() helper."""

    def test_non_pyod_module_returns_false(self):
        """Objects from non-pyod modules return False."""
        obj = MockProtocolCompliantDetector()
        assert _looks_like_pyod(obj) is False

    def test_builtin_type_returns_false(self):
        """Built-in types return False."""
        assert _looks_like_pyod([1, 2, 3]) is False
        assert _looks_like_pyod("string") is False
        assert _looks_like_pyod(42) is False


class TestAdaptWithActualPyOD:
    """Tests with actual PyOD detector (if available)."""

    @pytest.fixture
    def pyod_available(self):
        """Check if PyOD is available."""
        import importlib.util

        return importlib.util.find_spec("pyod") is not None

    def test_adapt_pyod_detector(self, pyod_available):
        """Real PyOD detector is adapted (either wrapped or passed through)."""
        if not pyod_available:
            pytest.skip("PyOD not installed")

        from pyod.models.iforest import IForest

        detector = IForest(n_estimators=10, random_state=42)
        adapted = adapt(detector)
        # PyOD detectors that already satisfy the protocol pass through unchanged
        # Those that don't get wrapped in PyODAdapter
        # Either way, the result should be protocol-compliant
        assert isinstance(adapted, AnomalyDetector)

    def test_adapted_pyod_is_protocol_compliant(self, pyod_available):
        """Adapted PyOD detector satisfies AnomalyDetector protocol."""
        if not pyod_available:
            pytest.skip("PyOD not installed")

        from pyod.models.iforest import IForest

        detector = IForest(n_estimators=10, random_state=42)
        adapted = adapt(detector)
        assert isinstance(adapted, AnomalyDetector)
