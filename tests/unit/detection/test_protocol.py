"""Unit tests for detection/protocol.py."""

from typing import Any, Self

import numpy as np

from nonconform.detection.protocol import AnomalyDetector


class CompliantDetector:
    """Detector that fully implements AnomalyDetector protocol."""

    def __init__(self):
        self._params = {"n_estimators": 100}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.standard_normal(len(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        return self


class MissingFit:
    """Detector missing fit method."""

    def decision_function(self, X):
        return np.array([1.0])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class MissingDecisionFunction:
    """Detector missing decision_function method."""

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class MissingGetParams:
    """Detector missing get_params method."""

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.array([1.0])

    def set_params(self, **params):
        return self


class MissingSetParams:
    """Detector missing set_params method."""

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.array([1.0])

    def get_params(self, deep=True):
        return {}


class TestProtocolCompliance:
    """Tests for AnomalyDetector protocol compliance checking."""

    def test_compliant_detector_passes_isinstance(self):
        """Compliant detector passes isinstance check."""
        detector = CompliantDetector()
        assert isinstance(detector, AnomalyDetector)

    def test_missing_fit_fails_isinstance(self):
        """Detector missing fit fails isinstance check."""
        detector = MissingFit()
        assert not isinstance(detector, AnomalyDetector)

    def test_missing_decision_function_fails_isinstance(self):
        """Detector missing decision_function fails isinstance check."""
        detector = MissingDecisionFunction()
        assert not isinstance(detector, AnomalyDetector)

    def test_missing_get_params_fails_isinstance(self):
        """Detector missing get_params fails isinstance check."""
        detector = MissingGetParams()
        assert not isinstance(detector, AnomalyDetector)

    def test_missing_set_params_fails_isinstance(self):
        """Detector missing set_params fails isinstance check."""
        detector = MissingSetParams()
        assert not isinstance(detector, AnomalyDetector)


class TestProtocolMethodSignatures:
    """Tests for protocol method signatures."""

    def test_fit_accepts_x_and_y(self):
        """fit() accepts X and optional y parameter."""
        detector = CompliantDetector()
        X = np.array([[1, 2], [3, 4]])

        # Should work without y
        result = detector.fit(X)
        assert result is detector

        # Should work with y=None
        result = detector.fit(X, y=None)
        assert result is detector

    def test_decision_function_returns_array(self):
        """decision_function() returns numpy array."""
        detector = CompliantDetector()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scores = detector.decision_function(X)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X)

    def test_get_params_returns_dict(self):
        """get_params() returns dictionary."""
        detector = CompliantDetector()
        params = detector.get_params()
        assert isinstance(params, dict)

    def test_get_params_deep_parameter(self):
        """get_params() accepts deep parameter."""
        detector = CompliantDetector()
        params_shallow = detector.get_params(deep=False)
        params_deep = detector.get_params(deep=True)
        assert isinstance(params_shallow, dict)
        assert isinstance(params_deep, dict)

    def test_set_params_returns_self(self):
        """set_params() returns detector instance."""
        detector = CompliantDetector()
        result = detector.set_params(n_estimators=50)
        assert result is detector

    def test_set_params_modifies_params(self):
        """set_params() actually modifies parameters."""
        detector = CompliantDetector()
        detector.set_params(new_param="value")
        params = detector.get_params()
        assert params.get("new_param") == "value"


class TestProtocolRuntimeCheckable:
    """Tests for @runtime_checkable decorator functionality."""

    def test_runtime_checkable_works(self):
        """Protocol is runtime checkable."""
        # This should not raise
        assert isinstance(CompliantDetector(), AnomalyDetector)

    def test_runtime_check_with_callable_attributes(self):
        """Protocol checks for callable methods."""

        # Create an object with non-callable attributes of the same names
        class FakeDetector:
            fit = "not a method"
            decision_function = "not a method"
            get_params = "not a method"
            set_params = "not a method"

        detector = FakeDetector()
        # Runtime checkable only checks for attribute existence, not callability
        # This is a known limitation of @runtime_checkable
        # The actual check happens at usage time
        assert isinstance(detector, AnomalyDetector)


class TestProtocolWithSklearnConventions:
    """Tests for sklearn-style detectors."""

    def test_sklearn_style_detector(self):
        """sklearn-style detector with clone() compatible interface."""

        class SklearnStyleDetector:
            def __init__(self, param1=1, param2="value"):
                self.param1 = param1
                self.param2 = param2

            def fit(self, X, y=None):
                return self

            def decision_function(self, X):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {"param1": self.param1, "param2": self.param2}

            def set_params(self, **params):
                for key, value in params.items():
                    setattr(self, key, value)
                return self

        detector = SklearnStyleDetector()
        assert isinstance(detector, AnomalyDetector)

        # Test sklearn conventions
        params = detector.get_params()
        assert "param1" in params
        assert "param2" in params

        detector.set_params(param1=42)
        assert detector.param1 == 42
