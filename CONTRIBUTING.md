# Contributing

Contributions are useful when they improve the library's correctness, scope,
usability, documentation, or interoperability.

Project-relevant contributions include bug reports, focused fixes, tests,
documentation corrections, detector compatibility notes, examples, and new
methods or approaches for conformal anomaly detection.

## Development Setup

Prerequisites:

- Python 3.12 or newer
- Git
- [uv](https://docs.astral.sh/uv/)

Clone and install:

```bash
git clone https://github.com/OliverHennhoefer/nonconform.git
cd nonconform
uv sync --group dev
uv pip install -e ".[all]"
```

Optional pre-commit setup:

```bash
uv run pre-commit install
```

## Before Opening a Pull Request

Keep changes focused. A pull request should have one clear purpose.

Run the checks that match your change:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pytest
```

If documentation under `docs/` changed, also run:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

When changes affect p-values, FDR control, weighting, calibration,
aggregation, metrics, or validity claims, include tests that cover the
statistical behavior and explain the before/after rationale in the pull request.

## API Compatibility

nonconform is a v1 project. Public APIs should remain stable unless a breaking
change is intentional and explicitly justified.

Treat these as public contracts:

- root exports in `nonconform.__all__`
- public module `__all__` exports
- documented constructor arguments, methods, properties, enum values, and
  dataclass fields

Implementation details under `nonconform._internal` are private.

## Issues

Bug reports should include:

- a minimal reproducible example
- expected and actual behavior
- Python, operating system, and dependency versions
- full traceback or warning text when relevant

Feature requests should describe the use case, the proposed method or approach,
the statistical or API impact, and any relevant literature or detector family
when applicable.
