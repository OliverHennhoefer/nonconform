from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Self

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest


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
    assert "Derandomized realized FDP:" in output
    assert "Derandomized realized true-positive rate:" in output
    detector = namespace["detector"]
    result = detector.last_selection_result
    assert result is not None
    assert result.n_repetitions == 5
    assert result.n_calibration == 1_000
    assert result.tie_seed is not None
    np.testing.assert_array_equal(namespace["decisions"], result.selected)
    assert detector.last_result is None


def test_fpr_bounds_notebook_smoke():
    example_path = Path(__file__).parents[2] / "examples" / "fpr_bounds.ipynb"
    notebook = json.loads(example_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, str(example_path), "exec"), namespace)

    assert namespace["IsolationForest"] is IsolationForest
    certificate = namespace["certificate"]
    threshold = namespace["threshold"]
    scores = namespace["scores"]
    mask = namespace["mask"]

    assert certificate.n_calibration == 1_000
    assert np.isfinite(threshold)
    assert certificate.bound_at(threshold) <= 0.05
    assert np.any(mask)
    assert np.all(mask[-5:])
    np.testing.assert_array_equal(mask, scores >= threshold)
    np.testing.assert_array_equal(mask, certificate.select(scores, threshold=threshold))


def test_fdp_notebook_smoke(monkeypatch, capsys):
    """Execute the real detector and certificate against a labeled fixture."""
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(3_000, 3))
    x_normal = rng.normal(size=(800, 3))
    x_anomaly = rng.normal(loc=5.0, size=(200, 3))
    x_test = np.vstack([x_normal, x_anomaly])
    y_test = np.r_[np.zeros(800, dtype=int), np.ones(200, dtype=int)]
    oddball = ModuleType("oddball")
    oddball.Dataset = SimpleNamespace(SHUTTLE="shuttle")
    oddball.load = lambda *args, **kwargs: (x_train, x_test, y_test)
    monkeypatch.setitem(sys.modules, "oddball", oddball)
    example_path = Path(__file__).parents[2] / "examples" / "fdp_bounds.ipynb"
    notebook = json.loads(example_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    logger = logging.getLogger("nonconform")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(
                    compile("".join(cell["source"]), str(example_path), "exec"),
                    namespace,
                )
    finally:
        logger.setLevel(original_level)
        logger.handlers[:] = original_handlers
    certificate = namespace["certificate"]
    assert certificate.n_calibration == 1_000
    assert certificate.n_test == 1_000
    assert certificate.method == "mc_thc"
    assert certificate.seed is None
    assert certificate.n_resamples == 1_000
    assert certificate.boost
    selected = certificate.select(0.01)
    assert np.all(selected[-200:])
    assert np.isfinite(certificate.bound_at(0.01))
    assert certificate.bound_at(0.01) < 0.5
    assert "Certified FDP upper bound:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "relative_path,n_calibration,n_anomalies",
    [
        ("README.md", 1_000, 20),
        ("docs/source/api/common_workflows.md", 1_000, 40),
        ("docs/source/examples/fdr_control.md", 1_000, 40),
        ("docs/source/user_guide/fdr_control.md", 1_500, 20),
    ],
)
def test_fdp_documentation_snippets(
    monkeypatch, relative_path, n_calibration, n_anomalies
):
    """Run the actual documented workflow under reproducible test sampling."""
    from nonconform._internal import fdp_bounds as core

    original = core._mc_summary_quantile

    def sampled(**kwargs):
        assert kwargs["seed"] is None
        return original(**{**kwargs, "seed": 123})

    monkeypatch.setattr(core, "_mc_summary_quantile", sampled)
    path = Path(__file__).parents[2] / relative_path
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    if relative_path == "README.md":
        code = next(
            block
            for block in blocks
            if "import numpy" in block and ".fit(x_train)" in block
        )
        construction = re.search(r"`(certificate = detector\.fdp_bounds[^`]+)`", text)
        assert construction is not None
        code += "\n" + construction.group(1) + "\ncutoff = 0.01\n"
    else:
        code = next(
            block
            for block in blocks
            if "import numpy" in block
            and "certificate =" in block
            and ".fdp_bounds(" in block
        )
    namespace = {"__name__": "__main__"}
    exec(compile(code, str(path), "exec"), namespace)
    certificate = namespace["certificate"]
    assert certificate.n_calibration == n_calibration
    assert certificate.seed is None
    assert certificate.boost
    assert certificate.n_resamples == 1_000
    assert namespace["cutoff"] == 0.01
    selected = certificate.select(namespace["cutoff"])
    assert np.all(selected[-n_anomalies:])
    bound = certificate.bound_at(namespace["cutoff"])
    assert np.isfinite(bound)
    assert bound < 0.5
    labels = np.r_[
        np.zeros(certificate.n_test - n_anomalies, dtype=int),
        np.ones(n_anomalies, dtype=int),
    ]
    realized = np.count_nonzero(selected & (labels == 0)) / np.count_nonzero(selected)
    assert realized <= bound
