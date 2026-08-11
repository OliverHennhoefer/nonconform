"""Local scaling benchmark for bulk sequential-rank priming.

Run manually from the repository root::

    uv run python benchmarks/benchmark_monitoring_prime.py

This script deliberately has no pass/fail timing threshold and is not part of
the test suite. It reports median elapsed time and adjacent size ratios so bulk
priming performance can be inspected without introducing timing-sensitive CI.
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

import numpy as np

from nonconform.monitoring import SequentialRankConformalizer


def _measure(size: int, repeats: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=size)
    elapsed: list[float] = []
    for _ in range(repeats):
        conformalizer = SequentialRankConformalizer(seed=seed)
        started = perf_counter()
        conformalizer.prime_many(samples)
        elapsed.append(perf_counter() - started)
    return median(elapsed)


def main() -> None:
    """Run the benchmark and print median timings with adjacent ratios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 20_000, 40_000]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timings = [(size, _measure(size, args.repeats, args.seed)) for size in args.sizes]
    previous: tuple[int, float] | None = None
    for size, elapsed in timings:
        ratio = "-"
        if previous is not None:
            previous_size, previous_elapsed = previous
            ratio = (
                f"{elapsed / previous_elapsed:.2f}x for {size / previous_size:.2f}x N"
            )
        print(f"{size:>10,} scores  {elapsed:.6f}s median  ratio={ratio}")
        previous = size, elapsed


if __name__ == "__main__":
    main()
