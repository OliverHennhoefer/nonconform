# Changelog

All notable changes to this project will be documented in this file (from `0.9.14+`).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.9.161 (Unreleased)

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
