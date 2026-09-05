"""Private typed provenance for detector-produced result snapshots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nonconform.structures import ConformalResult


class StrategyFamily(Enum):
    """Strategy families relevant to downstream statistical procedures."""

    SPLIT = auto()
    OTHER = auto()


class EstimationFamily(Enum):
    """Estimator families relevant to downstream statistical procedures."""

    EMPIRICAL = auto()
    CONDITIONAL_EMPIRICAL = auto()
    OTHER = auto()


class CalibrationMode(Enum):
    """Ways in which calibration scores can be produced."""

    INTEGRATED = auto()
    DETACHED = auto()


@dataclass(frozen=True, slots=True)
class BatchSignature:
    """Stable identity for one concrete input batch."""

    shape: tuple[int, ...]
    dtype: str
    digest: str


@dataclass(frozen=True, slots=True)
class ResultProvenance:
    """Typed facts needed to validate downstream result compatibility."""

    strategy_family: StrategyFamily
    estimation_family: EstimationFamily
    weighted: bool
    calibration_mode: CalibrationMode | None
    test_batch_signature: BatchSignature | None


def batch_signature(x: np.ndarray) -> BatchSignature:
    """Return a stable signature for a concrete batch."""
    contiguous = np.ascontiguousarray(x)
    values = contiguous
    if contiguous.dtype.hasobject:
        # Object-array bytes contain pointers. Hash values without losing integer
        # precision through a float conversion (e.g. nullable pandas Int64 inputs).
        values = pd.util.hash_array(contiguous.ravel(), categorize=False)
    digest = hashlib.blake2b(
        values.tobytes(),
        digest_size=16,
    ).hexdigest()
    return BatchSignature(
        shape=contiguous.shape,
        dtype=str(contiguous.dtype),
        digest=digest,
    )


def parse_result_provenance(
    result: ConformalResult,
    *,
    allow_legacy_metadata: bool,
) -> ResultProvenance | None:
    """Return native provenance or a compatibility view of legacy metadata."""
    native = result._provenance
    if native is not None:
        if not isinstance(native, ResultProvenance):
            raise ValueError("result contains invalid internal provenance.")
        return native
    if not allow_legacy_metadata:
        return None
    return _parse_legacy_metadata(result.metadata)


def _parse_legacy_metadata(metadata: Any) -> ResultProvenance | None:
    """Parse the released best-effort metadata contract for FDP compatibility."""
    if not isinstance(metadata, dict):
        return None
    if "kde" in metadata:
        return ResultProvenance(
            strategy_family=StrategyFamily.OTHER,
            estimation_family=EstimationFamily.OTHER,
            weighted=False,
            calibration_mode=None,
            test_batch_signature=None,
        )

    scope = metadata.get("nonconform")
    if scope is None:
        return None
    if not isinstance(scope, dict):
        raise ValueError("result.metadata['nonconform'] must be a dictionary.")

    return ResultProvenance(
        strategy_family=(
            StrategyFamily.SPLIT
            if scope.get("strategy") == "Split"
            else StrategyFamily.OTHER
        ),
        estimation_family=(
            EstimationFamily.EMPIRICAL
            if scope.get("estimation") == "Empirical"
            else EstimationFamily.OTHER
        ),
        weighted=bool(scope.get("weighted")),
        calibration_mode=None,
        test_batch_signature=None,
    )
