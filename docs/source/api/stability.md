# API Stability

nonconform 1.x treats public API stability as a release constraint.

## Stable Public Surface

The compatibility contract is defined by:

- `nonconform.__all__`, the intentionally small root import surface.
- `__all__` in public modules such as `nonconform.scoring`,
  `nonconform.resampling`, `nonconform.weighting`, `nonconform.fdr`,
  `nonconform.metrics`, `nonconform.martingales`, `nonconform.structures`,
  `nonconform.adapters`, and `nonconform.enums`.
- Public constructor parameters, public methods and properties, dataclass
  fields, enum members, and documented string literal values.

Symbols under `nonconform._internal` are private implementation details and are
not covered by the compatibility contract.

## Change Policy

Patch and minor releases should preserve documented public behavior. Prefer
additive APIs over changing or removing existing public symbols.

Breaking public API changes require a major-version release plan, release notes,
and documentation updates. Statistical-core behavior changes require explicit
before/after rationale because they can change validity claims even when the
Python signature is unchanged.

## Score Polarity Defaults

If `score_polarity` is omitted, the v1 default policy is:

- known scikit-learn normality-scoring detectors use `"higher_is_normal"`;
- PyOD detectors use `"higher_is_anomalous"`;
- custom detectors outside recognized families use `"higher_is_anomalous"`.

Use `score_polarity="auto"` when strict detector-family validation is desired;
it raises for custom detectors outside recognized PyOD and known scikit-learn
families.
