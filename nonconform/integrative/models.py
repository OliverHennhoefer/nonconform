"""Model specifications for integrative conformal detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

IntegrativeReference = Literal["inlier", "outlier"]
IntegrativeModelKind = Literal["one_class", "binary"]
BinaryScoreSource = Literal["auto", "predict_proba", "decision_function"]


@dataclass(slots=True)
class IntegrativeModel:
    """Typed model specification for the integrative detector.

    Attributes:
        kind: Model family, either ``"one_class"`` or ``"binary"``.
        estimator: sklearn-compatible or custom estimator instance.
        reference: Reference population for one-class models. Binary models use
            ``None`` because they can contribute to both the inlier and outlier
            preliminary p-value paths through score sign tuning.
        score_polarity: Optional score-polarity hint for one-class estimators.
            Accepted values match the public detector API.
        inlier_label: Class label representing inliers for binary estimators.
        score_source: Binary score extraction method.
        name: Optional human-readable label used in metadata/debug output.
    """

    kind: IntegrativeModelKind
    estimator: Any
    reference: IntegrativeReference | None = None
    score_polarity: (
        Literal["auto", "higher_is_anomalous", "higher_is_normal"] | None
    ) = None
    inlier_label: int | str | bool | None = None
    score_source: BinaryScoreSource = "auto"
    name: str | None = None

    @classmethod
    def one_class(
        cls,
        *,
        reference: IntegrativeReference,
        estimator: Any,
        score_polarity: Literal["auto", "higher_is_anomalous", "higher_is_normal"]
        | None = None,
        name: str | None = None,
    ) -> Self:
        """Create a one-class model specification."""
        if reference not in {"inlier", "outlier"}:
            raise ValueError(
                "reference must be either 'inlier' or 'outlier' for one-class models."
            )
        return cls(
            kind="one_class",
            estimator=estimator,
            reference=reference,
            score_polarity=score_polarity,
            name=name,
        )

    @classmethod
    def binary(
        cls,
        *,
        estimator: Any,
        inlier_label: int | str | bool = 0,
        score_source: BinaryScoreSource = "auto",
        name: str | None = None,
    ) -> Self:
        """Create a binary model specification."""
        if score_source not in {"auto", "predict_proba", "decision_function"}:
            raise ValueError(
                "score_source must be one of "
                "{'auto', 'predict_proba', 'decision_function'}."
            )
        return cls(
            kind="binary",
            estimator=estimator,
            reference=None,
            score_polarity=None,
            inlier_label=inlier_label,
            score_source=score_source,
            name=name,
        )

    def __post_init__(self) -> None:
        """Validate specification consistency."""
        if self.kind not in {"one_class", "binary"}:
            raise ValueError("kind must be either 'one_class' or 'binary'.")
        if self.kind == "one_class":
            if self.reference not in {"inlier", "outlier"}:
                raise ValueError(
                    "one_class model specifications require reference='inlier' "
                    "or reference='outlier'."
                )
            if self.score_source != "auto":
                raise ValueError(
                    "score_source is only applicable to binary model specifications."
                )
        if self.kind == "binary":
            if self.inlier_label is None:
                raise ValueError("binary model specifications require inlier_label.")
            if self.reference is not None:
                raise ValueError("binary model specifications do not accept reference.")


__all__ = [
    "BinaryScoreSource",
    "IntegrativeModel",
    "IntegrativeModelKind",
    "IntegrativeReference",
]
