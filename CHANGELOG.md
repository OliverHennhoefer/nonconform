# Changelog

All notable changes to this project will be documented in this file (from `0.9.14+`).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.95.0

### Added
- Change the interface to support the standard ``Empirical()`` as well as the new ``Probabilistic()`` approach.
  - ```Probabilistic()``` models the calibration set additionally via a suitable KDE to increase power in low-regime settings.
  - Structure and naming in the package was changed in this context.
- Adds ```BootstrapBaggedWeightEstimator``` for more stable weights in scenarios of extreme imbalance between calibration and test set size.
- Added ``ConformalResult`` snapshot object exposed via ``ConformalDetector.last_result`` to bundle p-values, scores, and weights.
- Added ```weighted_bh()``` for a simplistic weighted variant of the standard Benjamini-Hochberg procedure.

### Changed
- Migrate to pytest

## 0.91.0 (2025-10-09)

### Added
- Weighted Conformal Selection (``weighted_false_discovery_control()``) for rigorous FDR control when using weighted conformal p-values.

## 0.90.0 (2025-09-05)

### Changed
- Migration from ```sphinx``` to ``Material for MkDocs``.
  - Publication on GitHub Pages.
- Formatting with ``black`` was entirely replaced by ``ruff`` (e.g. pre-commits).
- Changed ``plus`` parameter defaults (``true``) and warning for manual override (to ``False``).
  - This is due to the lacking statistical guarantees if calibration scores and models are not aligned.
- Removed the `dec` parameter from `false_discovery_rate` and `statistical_power` to separate calculation from presentation.
- Refactored to follow a numpy-first approach for better efficiency and consistency
  - Further code quality improvements
- Unified `StandardConformalDetector` and `WeightedConformalDetector` into a single `ConformalDetector` class.
  - Standard conformal prediction: `ConformalDetector(detector=..., strategy=...)`
  - Weighted conformal prediction: `ConformalDetector(detector=..., strategy=..., weight_estimator=...)`
  - Automatic optimization: uses faster standard p-value calculation when no weight estimator is provided.
  - Maintains full backward compatibility with identical results for both standard and weighted modes.
- Refactored weight computation to use pluggable weight estimators.
  - Extracted weight computation logic into separate `BaseWeightEstimator` classes.
  - Added `LogisticWeightEstimator`, `ForestWeightEstimator`, and `IdentityWeightEstimator`.

### Fixed
- Bug Fix in ```JackknifeBootstrap()``` regarding the original plus-variant.

## 0.9.200 (2025-08-28)

### Changed
- Refactored dataset loading module with unified `load(name: Dataset)` API and DatasetManager architecture.
- Migrated from `.parquet.gz` to `.npz` format for improved performance.
- Added dataset registry system with `DatasetInfo` dataclass and discovery functions `list_available()` and `get_info()`.
- Extended the set of available datasets

## 0.9.168/0.9.169 (2025-08-26)

### Fixed
- Reupload.

## 0.9.167 (2025-08-26)

### Changed
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
