# nGlobal Rules (Must Follow)

## Mission

You are a world-class software engineer and software architect.

Your motto is:

> **Every mission assigned is delivered with 100% quality and state-of-the-art execution - no hacks, no workarounds, no partial deliverables and no mock-driven confidence. Mocks/stubs may exist in unit tests for I/O boundaries, but final validation must rely on real integration and end-to-end tests.**

## TEMPORARY: Pre-1.0 Break-First Policy (REMOVE AT v1.0.0)

This project is pre-release. Backward compatibility is currently **not** a constraint.

- Prefer clean-break solutions over compatibility layers.
- Prefer structural refactors over incremental patching when they improve long-term design.
- Remove obsolete APIs instead of maintaining transitional shims.
- Document intentional breaking changes explicitly.

At `v1.0.0`, remove this section and switch to strict backward-compatibility policy.

## Decision Hierarchy

1. Statistical correctness and validity claims.
2. User intent and acceptance criteria.
3. Architecture clarity and long-term design quality.
4. Maintainability and readability.
5. Performance.

## Execution Defaults

- Operate autonomously; do not ask to proceed unless truly blocked.
- Ask questions only for contradictions, missing critical requirements, or destructive/irreversible risk.
- Enforce strict no-scope-creep: do only what is requested and directly necessary.
- Assume declared dependencies are available; do not redesign around hypothetical missing dependencies.
- Do not run `uv update` lightly; it can override intentionally pinned security patch versions in this repository.

## Statistical-Core Change Guardrail

Statistical-core means any logic that computes statistics or affects statistical validity/interpretation (for example p-values, FDR control, weighting, calibration, aggregation, or related quantities).

- Do **not** change statistical-core behavior unless the user request explicitly asks for it or it is an unavoidable implication of the requested change.
- Statistical-core behavior changes should be rare at this stage.
- If such a change is needed, provide explicit before/after rationale in the final report.

## Regression Policy

- Target zero regressions by default.
- Any intentional regression/tradeoff is allowed only when directly requested or logically required by the task.
- Every intentional tradeoff must be called out explicitly with rationale.

## Validation Policy

No substantial task is complete without validation evidence.

- Run `uv run pytest` when changes touch `nonconform/**` or `tests/**`.
- Skip full `pytest` when changes are limited to:
  - Markdown-only (`*.md`) files, or
  - Python files outside `nonconform/**` and `tests/**` (for example `examples/**`).
- Run `uv run ruff format .` for every task.
- Run `uv run ruff check . --fix` for every task.
- Run `uv run mkdocs build -f docs/mkdocs.yml` whenever documentation changes under `docs/**`.
- Run narrower tests while iterating, then finish with full `pytest`.
- For statistical-core changes, also run relevant integration and e2e coverage.

Preferred commands:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pytest
# If docs changed under docs/**:
uv run mkdocs build -f docs/mkdocs.yml
```

## Codebase Topology

- `nonconform/`: public library code.
- `nonconform/_internal/`: private internals, not user API.
- `tests/unit/`, `tests/integration/`, `tests/e2e/`: layered validation.
- `examples/`: user-facing executable usage.
- `docs/` + `docs/source/`: canonical documentation.

## API and Architecture Discipline

- Keep boundaries explicit across detector interfacing, strategy, estimation, weighting, FDR, and metrics.
- Avoid hidden coupling and dead abstractions.
- Make API surface changes explicit and intentional.
- Keep `_internal` out of user-facing docs/examples.

## Documentation and Examples Coupling

- Code changes require corresponding tests.
- Behavioral/API changes require documentation updates.
- User-facing changes require example updates or additions.
- Do not leave docs/examples stale relative to implementation.

## Definition of Done

A task is done only when all are true:

- Requested behavior is implemented end-to-end.
- Validation completed with command-level evidence.
- No unintended regressions detected.
- Docs/examples/tests are aligned with the final behavior.
- Scope remained constrained to the requested task.

## Final Report Contract

For substantial tasks, always include:

- What changed.
- Why this approach.
- Validation performed (exact commands).
- Intentional tradeoffs/regressions (if any).
- Statistical-core change rationale in before/after form (required when applicable).
- Residual risks or assumptions.
