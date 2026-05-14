# Contributing

Contributions should improve the library's correctness, scope, usability,
documentation, or interoperability.

Useful contributions include focused bug fixes, tests, documentation
corrections, detector compatibility notes, examples, and new methods or
approaches for conformal anomaly detection, conformal inference, FDR control,
covariate-shift workflows, or related detector interfaces.

## Setup

```bash
git clone https://github.com/OliverHennhoefer/nonconform.git
cd nonconform
uv sync --group dev
uv pip install -e ".[all]"
```

Optional pre-commit hooks:

```bash
uv run pre-commit install
```

## Checks

Run the checks that match the change:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pytest
```

For documentation changes:

```bash
uv run mkdocs build -f docs/mkdocs.yml
```

## Pull Requests

Keep pull requests focused and include:

- what changed and why
- exact validation commands run
- public API impact, if any
- statistical-core impact, if any
- relevant papers, detector libraries, or implementation references when useful

Changes to p-values, FDR control, weighting, calibration, aggregation, metrics,
or validity claims need tests and an explicit before/after rationale.

## API Stability

nonconform is a v1 project. Public APIs should remain stable unless a breaking
change is intentional and documented.

Public contracts include root exports, public module `__all__` exports,
documented constructor arguments, methods, properties, enum values, and
dataclass fields. `nonconform._internal` is private.

## Issues

Bug reports should include a minimal reproducible example, expected and actual
behavior, environment details, and tracebacks or warnings when relevant.

Feature requests should describe the workflow, proposed method or approach,
statistical or API impact, and relevant prior work when applicable.
