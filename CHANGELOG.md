# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `DerandomizedSplits` for repeated split-conformal e-value selection
  through `ConformalDetector.fit()` and `select()`, with automatic randomized
  tie handling and defensive diagnostics in `last_selection_result`. Existing
  strategies and standalone e-value functions retain their behavior.
- Added result-aware derandomized conformal e-value selection through
  `nonconform.fdr.select_conformal_e_values`, plus low-level e-value and e-BH
  primitives. For unmodified result snapshots, the workflow verifies
  split-conformal provenance and exact test batch identity, rejects score ties
  by default, and supports reproducible randomized tie-breaking through `tie_seed`.

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
