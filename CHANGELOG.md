# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `nonconform.fdr.FPRCertificate` with immutable simultaneous raw-score
  false-positive-rate bounds, target-FPR threshold inversion, and native
  `ConformalDetector.fpr_bounds()` / `ConformalResult.fpr_bounds()` integration
  for unweighted `Split` calibration independently of the p-value estimator.
  A deterministic finite-sample one-sided KS band covers the population inlier
  false-alarm curve with the requested probability over the calibration draw,
  conditional on a scoring map fixed independently of calibration. Coverage
  requires clean, i.i.d. calibration and future inlier scores from the same
  distribution. Returned thresholds preserve NumPy precision
  when compared with lower-precision score arrays.
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

- **Breaking: FDP certification API replacement.** Use
  `ConformalDetector.fdp_bounds(x, ...)`, `ConformalResult.fdp_bounds(...)`, or
  `nonconform.fdr.FDPCertificate.from_p_values(...)`. Certificates are immutable
  and prepare threshold queries once. Move `thresholds=` from construction to
  `to_frame(thresholds=...)` or `bound_at(...)`; omit inapplicable method options.
  Native entry points expose only `confidence` and use the default boosted
  Monte Carlo THC envelope; advanced settings belong to the expert factory.
  Berk-Jones numerical tolerance is internal. Monte Carlo confidence covers
  data and independent envelope sampling jointly, not every fixed seed or
  realized envelope; settings must be chosen before inspecting bounds.
  Native entry points require unmodified native empirical Split/detached,
  unweighted snapshots. External p-values use the explicit expert factory.
  This intentional API break is limited to FDP certification; other workflows
  are unchanged. Replacement of the released FDP helpers is planned for the
  next major release; no version metadata is bumped in this change. The
  certificate factories and their simplified signatures are unreleased.

### Removed

- Removed `FDPBoundResult`, `conformal_fdp_upper_bound`, and
  `conformal_fdp_upper_bound_from_result` without compatibility aliases.

### Fixed

- FDP higher-criticism shape parameters now reject `beta > 1`, outside the
  range supported by the jump-point statistic calculation. Berk-Jones
  inversion now returns conservative lower endpoints, includes p-values equal
  to those endpoints, handles unreachable lower roots, and terminates when
  floating-point bisection stalls. Previously,
  accepted expert settings could understate FDP bounds or hang construction;
  Berk-Jones bounds can increase to preserve coverage.
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
