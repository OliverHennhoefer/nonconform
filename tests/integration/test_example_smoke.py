from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Self

import numpy as np


class FakeIForest:
    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        _ = X, y
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.sum(np.asarray(X, dtype=float) ** 2, axis=1)

    def get_params(self, deep: bool = True) -> dict[str, int | None]:
        _ = deep
        return {"random_state": self.random_state}

    def set_params(self, **params: int | None) -> Self:
        if "random_state" in params:
            self.random_state = params["random_state"]
        return self


def test_derandomized_e_values_notebook_smoke(monkeypatch, capsys):
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(1_200, 3))
    x_test_normal = rng.normal(size=(20, 3))
    x_test_anomaly = rng.normal(loc=4.0, size=(5, 3))
    x_test = np.vstack([x_test_normal, x_test_anomaly])
    y_test = np.array([0] * len(x_test_normal) + [1] * len(x_test_anomaly))

    oddball = ModuleType("oddball")
    oddball.Dataset = SimpleNamespace(SHUTTLE="shuttle")
    oddball.load = lambda *args, **kwargs: (x_train, x_test, y_test)

    pyod = ModuleType("pyod")
    pyod_models = ModuleType("pyod.models")
    pyod_iforest = ModuleType("pyod.models.iforest")
    pyod_iforest.IForest = FakeIForest

    monkeypatch.setitem(sys.modules, "oddball", oddball)
    monkeypatch.setitem(sys.modules, "pyod", pyod)
    monkeypatch.setitem(sys.modules, "pyod.models", pyod_models)
    monkeypatch.setitem(sys.modules, "pyod.models.iforest", pyod_iforest)

    example_path = (
        Path(__file__).parents[2] / "examples" / "derandomized_e_values.ipynb"
    )
    notebook = json.loads(example_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    root_logger = logging.getLogger("nonconform")
    original_level = root_logger.level
    original_handlers = list(root_logger.handlers)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, str(example_path), "exec"), namespace)
    finally:
        root_logger.setLevel(original_level)
        root_logger.handlers[:] = original_handlers

    output = capsys.readouterr().out
    assert "Derandomized discoveries:" in output
    assert "Empirical FDR:" in output
    assert "Empirical Power:" in output
