# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Standardized `weighted_bh` to support the same input conventions as
  `weighted_false_discovery_control` (`result` and/or raw arrays).
- Added a `seed` parameter to `weighted_bh` for reproducible internal
  weighted p-value recomputation.
- Replaced fit-gated `RuntimeError` checks with
  `sklearn.exceptions.NotFittedError` in detector and weight-estimator APIs.

### Breaking

- Renamed `BootstrapBaggedWeightEstimator` constructor parameter from
  `n_bootstrap` to `n_bootstraps` (no backward-compatible alias).

## [1.0.0] - YYYY-MM-DD

### Added

### Changed

### Fixed
