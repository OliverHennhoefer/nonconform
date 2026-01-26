"""Deterministic seed utilities."""

from __future__ import annotations

import numpy as np


def derive_seed(*parts: int) -> int:
    """Derive a stable 32-bit seed from one or more integer parts."""
    if not parts:
        raise ValueError("At least one seed part is required.")

    normalized = [int(part) & 0xFFFFFFFF for part in parts]
    seq = np.random.SeedSequence(normalized)
    return int(seq.generate_state(1)[0])
