"""Private typed provenance for detector-produced result snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

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
    # Empirical tie handling; None for non-empirical estimators.
    tie_break: TieBreakMode | None = None
    # Group-conditional (Mondrian) calibration.
    grouped: bool = False

    def __setstate__(self, state: Sequence[Any]) -> None:
        """Restore pickles, including 2.0 ones written before ``grouped``."""
        restore_fields(self, state, grouped=False)


def restore_fields(instance: Any, state: Sequence[Any], **defaults: Any) -> None:
    """Restore a slotted frozen dataclass; default fields that old pickles lack."""
    names = [item.name for item in fields(instance)]
    values = {**defaults, **dict(zip(names, state, strict=False))}
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def batch_signature(x: Any) -> BatchSignature:
    """Return a stable signature for a concrete batch.

    A pandas DataFrame is hashed by its values.
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    contiguous = np.ascontiguousarray(x)
    hasher = hashlib.blake2b(digest_size=16)
    if contiguous.dtype.hasobject:
        # Object-array bytes contain pointers. Hash values without losing integer
        # precision through a float conversion (e.g. nullable pandas Int64 inputs).
        flat = contiguous.ravel()
        hasher.update(pd.util.hash_array(flat, categorize=False).tobytes())
        # pandas hashes mixed objects by their string form, so 1 and "1" would
        # collide. Bind each element's type to keep such batches distinct.
        element_types = np.array(
            [f"{type(v).__module__}.{type(v).__qualname__}" for v in flat],
            dtype=object,
        )
        hasher.update(pd.util.hash_array(element_types, categorize=True).tobytes())
    else:
        hasher.update(contiguous.tobytes())
    digest = hasher.hexdigest()
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
