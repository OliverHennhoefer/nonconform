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


def test_one_class_model_rejects_binary_only_fields_on_direct_construction():
    with pytest.raises(ValueError, match="inlier_label is only applicable to binary"):
        IntegrativeModel(
            kind="one_class",
            estimator=object(),
            reference="inlier",
            inlier_label=1,
        )


def test_binary_model_rejects_one_class_only_fields_on_direct_construction():
    with pytest.raises(
        ValueError,
        match="score_polarity is only applicable to one_class",
    ):
        IntegrativeModel(
            kind="binary",
            estimator=object(),
            inlier_label=0,
            score_polarity="higher_is_anomalous",
        )
