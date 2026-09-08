---
description: "Compatibility policy for nonconform v1 public APIs, exports, documentation, and statistical validity claims."
---

# API Stability

nonconform 1.x treats public API stability as a release constraint.

## Stable Public Surface

The compatibility contract is defined by:

- `nonconform.__all__`, the intentionally small root import surface.
- `__all__` in public modules such as `nonconform.scoring`,
  `nonconform.resampling`, `nonconform.weighting`, `nonconform.fdr`,
  `nonconform.metrics`, `nonconform.martingales`, `nonconform.monitoring`,
  `nonconform.structures`, `nonconform.adapters`, and `nonconform.enums`.
- Public constructor parameters, public methods and properties, dataclass
  fields, enum members, and documented string literal values.

Internal implementation symbols are private details and are
not covered by the compatibility contract.

## Change Policy

Patch and minor releases should preserve documented public behavior. Prefer
additive APIs over changing or removing existing public symbols.

Breaking public API changes require a major-version release plan, release notes,
and documentation updates. Statistical-core behavior changes require explicit
before/after rationale because they can change validity claims even when the
Python signature is unchanged.

## Planned 2.0 FDP certification changes

The Unreleased FDP certification restrictions target **2.0** because previously
accepted configurations now raise errors: THC requires `0 < beta <= 1`, native
snapshots require a recognized empirical tie mode recorded at computation, and
randomized certification requires finite stored test scores without exact
calibration/test score ties. Classical certification continues to support tied
scores, and numerical bounds for supported configurations are unchanged.

`FDPCertificate.threshold_for(max_fdp=...)` is additive. The stricter checks are
limited to certification; p-value computation and ordinary selection retain
their behavior. See the [FDP migration guide](../user_guide/fdr_control.md#migration-for-20-certificate-restrictions)
for migration steps and statistical rationale. Version bump and publication
are separate release work.

## Score Polarity Defaults

If `score_polarity` is omitted, the v1 default policy is:

- known scikit-learn normality-scoring detectors use `"higher_is_normal"`;
- PyOD detectors use `"higher_is_anomalous"`;
- custom detectors outside recognized families use `"higher_is_anomalous"`.

Use `score_polarity="auto"` when strict detector-family validation is desired;
it raises for custom detectors outside recognized PyOD and known scikit-learn
families.
