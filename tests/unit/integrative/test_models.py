import pytest

from nonconform import IntegrativeModel


def test_one_class_model_requires_valid_reference():
    with pytest.raises(ValueError, match="reference must be either"):
        IntegrativeModel.one_class(reference="bad", estimator=object())  # type: ignore[arg-type]


def test_binary_model_rejects_invalid_score_source():
    with pytest.raises(ValueError, match="score_source must be one of"):
        IntegrativeModel.binary(
            estimator=object(),
            score_source="margin",  # type: ignore[arg-type]
        )


def test_binary_model_defaults_are_valid():
    model = IntegrativeModel.binary(estimator=object())
    assert model.kind == "binary"
    assert model.inlier_label == 0
    assert model.score_source == "auto"
