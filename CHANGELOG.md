# Changelog

All notable changes to this project will be documented in this file (from `0.9.14+`).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.9.168 (2025-08-26)

### Fixed
- Reupload.

## 0.9.167 (2025-08-26)

### Changed
- Consolidated README files by removing separate PyPI version and using conditional sections for platform-specific content.
- Merged `dev` and `docs` optional dependencies into a single `dev` group for simplified dependency management.

## 0.9.166 (2025-08-26)

### Added
- Warning system for `BatchGenerator` when small anomaly proportions truncate to zero anomalies per batch.
  - Users receive actionable guidance suggesting minimum batch size or probabilistic mode.
- Test coverage for small anomaly proportions (0.5%, 0.25%) across all generator modes.
  - Validates exact proportion handling in both proportional and probabilistic modes.

### Changed
- Simplified logging system to use standard Python logging conventions.
  - Default INFO level shows warnings and errors by default.
  - Users can control verbosity with `logging.getLogger("nonconform").setLevel(level)`.
  - Progress bars (tqdm) remain always visible regardless of logging level.
- "Aggregating models" progress bars now only appear at DEBUG level to reduce verbosity during inference.

## 0.9.165 (2025-08-26)

### Changed
- ``JackknifeBootstrap()`` uses now vectorized operations for the calibration procedure.

## 0.9.164 (2025-08-21)

### Fixed
- Bug fix in ```WeightedConformalDetector()```.
  - Adjusted test cases.

### Changed
- Minor code changes.

## 0.9.163 (2025-08-21)

### Changed
- The strategies ``Bootstrap()`` and ``Randomized()`` (i.e. randomized leave-p-out) are now structured into the sub dir ``experimental``.
    - The methods were moved as they are statistically more inefficient than the 'classical' methods.
    - The methods parameters ``plus`` default value was set to `True`, to guarantee a minimum of statistical validity as the guarantee otherwise does not hold.
      - Users will receive a warning if ``plus`` is manually set to `False`
- The test coverage was extended.
  - The test folder structure was optimised for higher granularity.

### Added
- The Jackknife+-after-Bootstrap was added as ```JackknifeBootstrap()``` (Kim et al., 2020).

### Changed
- Standardized parameter name from `random_state` to `seed` across all nonconform classes and functions for consistency.
  - Affects data loading functions (`load_*`), data generators (`BatchGenerator`, `OnlineGenerator`), and their base classes.

### Fixed
- After recent rework of the reproducibility approach, now also the ``load()`` method for all built-in dataset are truly random by default (for ``setup=True``).

## 0.9.162 (2025-08-20)

### Fixed
- Resolved module name conflict where `nonconform.utils.func.logging` shadowed Python's standard `logging` module, causing `AttributeError` when using `logging.basicConfig()`

## 0.9.161 (2025-08-19)

### Added
- The new strategy ``Randomized()`` implements randomized leave-p-out (rLpO) to interpolate between existing strategies.

### Changed
- The approach to reproducibility was reworked to allow true randomness when no ``seed`` is provided in the main classes.
  - Previously, the seed was internally set to 1, preventing truly random behavior.
- Removes ``silent`` parameter from ``ExtremeConformalDetector()``, ``StandardConformalDetector()`` and ``WeightedConformalDetector()``.
  - The parameter is being replaced by more consistent logging-based progress control.
  - Documentation was updated and an example for logging configuration was added in ``examples/utils/``.
- Centralized version handling with ``nonconform/__init__.py`` as single source of truth.
- Reworked `README.md` to reflect the current scope of features.
- Minor code refinements.

## 0.9.15 (2025-08-13)

### Added
- Callback for `Bootstrap()` strategy to inspect the calibration set.
  - Mainly for research purposes, this feature may monitor calibration set convergence and inform early stopping.
  - Respective usage example was added, documentation was updated accordingly.

### Changed
- Simplified building the documentation on Linux (`.docs/Makefile`) and Windows (`./docs/make.bat`).
  - On Windows, `.\make.bat` compiles to `.html`, on Linux/WSL `.\make.bat pdf` compiles to `.pdf`.
    - Mind the `[docs]` additional dependency.

### Security
- Migration to `numpy 2.x.x`
