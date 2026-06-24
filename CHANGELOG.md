# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/OliverHennhoefer/nonconform/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/OliverHennhoefer/nonconform/compare/v1.0.0...v1.0.1
