# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `FDPCertificate.threshold_for(max_fdp=...)` to maximize discoveries
  among observed p-value cutoffs whose simultaneous FDP bound meets a target.
  It returns `None` when no nonempty selection qualifies and reuses certificate
  state without rescoring or Monte Carlo resampling.
- Added `nonconform.martingales.MixtureMartingale` for fixed-weight composition
  of independently maintained martingale, harmonic restart, CUSUM, and
  Shiryaev-Roberts evidence, including direct `ExchangeabilityMonitor` support.
  Existing `SimpleMixtureMartingale` behavior is unchanged.
- Added `DerandomizedSplits` for repeated split-conformal e-value selection
  through `ConformalDetector.fit()` and `select()`, with automatic randomized
  tie handling and defensive diagnostics in `last_selection_result`. Existing
  strategies and standalone e-value functions retain their behavior.
- Added result-aware derandomized conformal e-value selection through
  `nonconform.fdr.select_conformal_e_values`, plus low-level e-value and e-BH
  primitives. For unmodified result snapshots, the workflow verifies
  split-conformal provenance and exact test batch identity, rejects score ties
  by default, and supports reproducible randomized tie-breaking through `tie_seed`.

### Changed

- **Breaking, targeting 2.0: supported FDP certificate configurations.** THC
  now requires `0 < beta <= 1`. Native certification requires immutable recorded
  empirical tie-mode provenance; randomized certification additionally requires
  finite stored test scores and rejects exact calibration/test score ties.
  Classical tied-score certification remains supported, including snapshots
  without test scores. Choose classical tie handling before computing p-values
  for discrete scoring rules, and recompute snapshots missing tie provenance.
  Expert `from_p_values(...)` callers remain responsible for the supported score
  construction. These checks replace previously accepted unsupported
  configurations; supported certificate numerics, p-value computation, and
  ordinary selection are unchanged. See the FDP guide for migration and
  statistical rationale. No version bump or publication is included.
- **Breaking: FDP certification API replacement.** Use
  `ConformalDetector.fdp_bounds(x, ...)`, `ConformalResult.fdp_bounds(...)`, or
  `nonconform.fdr.FDPCertificate.from_p_values(...)`. Certificates are immutable
  and prepare threshold queries once. Move `thresholds=` from construction to
  `to_frame(thresholds=...)` or `bound_at(...)`; omit inapplicable method options.
  Native entry points require unmodified native empirical Split/detached,
  unweighted snapshots. External p-values use the explicit expert factory.
  This intentional API break is limited to FDP certification; other workflows
  and finite numerical reference outputs are unchanged. No version bump is
  included in this change.

### Removed

- Removed `FDPBoundResult`, `conformal_fdp_upper_bound`, and
  `conformal_fdp_upper_bound_from_result` without compatibility aliases.

### Fixed

- HC certificates with a positive infinite Monte Carlo cutoff now use a
  conservative unit ECDF envelope instead of producing endpoint `NaN`s:
  nonempty selections have FDP bound 1, empty selections have bound 0.
  Sampling, the quantile convention, and finite-cutoff formulas are unchanged.

## [1.1.1] - 2026-08-19

### Changed

- Classical weighted empirical p-values include calibration mass tied with
  the test score, aligning deterministic weighted and unweighted empirical
  p-values for discrete scores.
- Seeded randomized WCS pruning uses a dedicated deterministic random stream
  instead of replaying the empirical p-value stream.

## [1.1.0] - 2026-08-11

### Added

- Added `nonconform.monitoring` with `SequentialRankConformalizer`,
  `ExchangeabilityMonitor`, and `MonitorState` for frozen-score sequential
  conformal ranks and end-to-end exchangeability monitoring with privately
  owned sequential state and atomic reference priming.
- Added `ExchangeabilityMonitor.from_split_detector(...)` as an additive bridge
  from existing fitted, unweighted `Split` detectors without changing their
  fixed-calibration p-value behavior.
- Added stepwise `e_value` and `log_e_value` fields to `MartingaleState`.

## [1.0.2] - 2026-08-01

### Added

- Added `ConformalDetector.compute_p_value` for explicit single-observation
  p-value computation in standard conformal streaming workflows.
- Added post-hoc FDP upper bounds for unweighted conformal p-values via
  `nonconform.fdr.conformal_fdp_upper_bound`, including certified precision
  lower bounds and envelope methods `mc_thc`, `mc_hc`, `mc_ks`, `ks`, and
  `mc_bj`.
### Changed

- Restricted cached-result FDP certificates to known supported empirical split
  conformal scopes.

## [1.0.1] - 2026-05-20

### Security

- Bumped indirect dependency `idna` from `3.10` to `3.15`.
- Bumped indirect dependency `pymdown-extensions` from `10.16.1` to `10.21.3`.

[Unreleased]: https://github.com/OliverHennhoefer/nonconform/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/OliverHennhoefer/nonconform/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/OliverHennhoefer/nonconform/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/OliverHennhoefer/nonconform/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/OliverHennhoefer/nonconform/compare/v1.0.0...v1.0.1
