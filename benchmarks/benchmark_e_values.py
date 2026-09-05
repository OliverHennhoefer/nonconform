"""Reproduce conformal e-value core timings against a slow reference.

This benchmark is intentionally standalone and excluded from timing-sensitive CI.
Run it from the repository root with::

    uv run python benchmarks/benchmark_e_values.py
"""

from __future__ import annotations

import argparse
import platform
from collections.abc import Callable
from time import perf_counter

import numpy as np

from nonconform.fdr import conformal_e_values

DEFAULT_SIZES = (10_000, 20_000, 50_000)
DEFAULT_N_CALIBRATION = 200
DEFAULT_ALPHA_BH = 0.1
DEFAULT_SEED = 2026


def slow_reference(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha_bh: float,
) -> np.ndarray:
    """Compute one split's e-values by scanning every candidate threshold."""
    threshold = float("inf")
    for candidate in np.sort(np.concatenate((test_scores, calib_scores))):
        n_test_above = int(np.count_nonzero(test_scores >= candidate))
        if n_test_above == 0:
            continue
        n_calib_above = int(np.count_nonzero(calib_scores >= candidate))
        estimated_fdp = (test_scores.size / calib_scores.size) * (
            n_calib_above / n_test_above
        )
        if estimated_fdp <= alpha_bh:
            threshold = float(candidate)
            break

    n_calib_above = int(np.count_nonzero(calib_scores >= threshold))
    evidence = (calib_scores.size + 1) / (n_calib_above + 1)
    return evidence * (test_scores >= threshold)


def tie_free_scores(
    n_test: int,
    n_calibration: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed, globally unique calibration and test scores."""
    rng = np.random.default_rng(seed)
    scores = rng.permutation(n_test + n_calibration).astype(float)
    return scores[:n_test], scores[n_test:]


def timed(callable_: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    """Run a callable once and return its result and wall-clock duration."""
    started = perf_counter()
    result = callable_()
    return result, perf_counter() - started


def benchmark(
    sizes: tuple[int, ...],
    *,
    n_calibration: int,
    alpha_bh: float,
    seed: int,
) -> None:
    """Run reference/current comparisons and print timings."""
    print("Conformal e-value benchmark")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(
        "Configuration: "
        f"n_test={sizes}, n_calibration={n_calibration}, "
        f"repetitions=1, alpha_bh={alpha_bh}, seed={seed}, ties=none"
    )
    print("size\treference_s\tcurrent_s\tspeedup")

    for n_test in sizes:
        test_scores, calib_scores = tie_free_scores(
            n_test,
            n_calibration,
            seed=seed + n_test,
        )
        expected, reference_seconds = timed(
            lambda: slow_reference(
                test_scores,
                calib_scores,
                alpha_bh=alpha_bh,
            )
        )
        actual, current_seconds = timed(
            lambda: conformal_e_values(
                test_scores,
                calib_scores,
                alpha_bh=alpha_bh,
            )
        )
        np.testing.assert_array_equal(actual, expected)
        speedup = reference_seconds / current_seconds
        print(
            f"{n_test}\t{reference_seconds:.6f}\t{current_seconds:.6f}\t{speedup:.2f}x"
        )


def parse_args() -> argparse.Namespace:
    """Parse optional benchmark configuration overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--n-calibration", type=int, default=DEFAULT_N_CALIBRATION)
    parser.add_argument("--alpha-bh", type=float, default=DEFAULT_ALPHA_BH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark from command-line arguments."""
    args = parse_args()
    benchmark(
        tuple(args.sizes),
        n_calibration=args.n_calibration,
        alpha_bh=args.alpha_bh,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
