import numpy as np

from nonconform.integrative._core import self_inclusive_ranks


def test_self_inclusive_ranks_accepts_empty_scores() -> None:
    result = self_inclusive_ranks(np.array([], dtype=float))

    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (0,)
