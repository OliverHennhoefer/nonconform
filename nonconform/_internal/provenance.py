"""Private typed provenance for detector-produced result snapshots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .constants import TieBreakMode

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
    empirical_tie_break: TieBreakMode | None = None


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


def parse_result_provenance(result: ConformalResult) -> ResultProvenance | None:
    """Return native provenance; manually populated metadata is not trusted."""
    native = result._provenance
    if native is not None and not isinstance(native, ResultProvenance):
        raise ValueError("result contains invalid internal provenance.")
    return native
