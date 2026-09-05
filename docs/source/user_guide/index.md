---
description: "Navigate nonconform concepts, batch discovery control, weighted conformal inference, sequential monitoring, evaluation, and API guidance."
---

# User guide

`nonconform` supports two primary workflows built on anomaly scores:

- **Batch discovery control:** calibrate p-values for a fixed family and select
  anomalies with a justified FDR procedure.
- **Sequential change monitoring:** generate randomized sequential conformal
  p-values, accumulate evidence with a martingale, and trigger configured
  alarms.

Choose the workflow first. Their p-values, error targets, and evaluation metrics
are different.

## Start from your task

| Task | Read first | Then |
|---|---|---|
| Select anomalies in one batch | [Conformal inference](conformal_inference.md) | [FDR control](fdr_control.md) |
| Stabilize repeated split selections | [Derandomized e-values](fdr_control.md#derandomized-conformal-e-values) | [Batch evaluation](batch_evaluation.md) |
| Monitor an ordered stream for change | [Exchangeability martingales](exchangeability_martingales.md) | [Streaming evaluation](streaming_evaluation.md) |
| Handle modeled covariate shift | [Weighted conformal](weighted_conformal.md) | [FDR control](fdr_control.md#weighted-conformalized-selection) |
| Choose split, CV, jackknife, or bootstrap | [Conformalization strategies](conformalization_strategies.md) | [Choosing strategies](choosing_strategies.md) |
| Integrate a detector | [Detector compatibility](detector_compatibility.md) | [Common workflows](../api/common_workflows.md) |
| Diagnose a failure | [Troubleshooting](troubleshooting.md) | [Input validation](input_validation.md) |

## Foundations

| Page | Purpose |
|---|---|
| [Statistical concepts](statistical_concepts.md) | Short definitions of p-values, exchangeability, FDR, power, covariate shift, and Ville control |
| [Conformal inference](conformal_inference.md) | Rank construction, data roles, marginal and conditional validity, ties, and batch versus sequential p-values |
| [Conformalization strategies](conformalization_strategies.md) | Exact mechanics and statistical scope of `Split`, `CrossValidation`, jackknife, and bootstrap |
| [Choosing strategies](choosing_strategies.md) | Decision process based on validity needs, resolution, model-fit budget, and empirical evaluation |

## Applied workflows

| Page | Purpose |
|---|---|
| [Detector compatibility](detector_compatibility.md) | scikit-learn, PyOD, custom protocols, blocked batch-adaptive models, and score polarity |
| [Weighted conformal](weighted_conformal.md) | Covariate-shift assumptions, density-ratio estimators, weight diagnostics, and WCS |
| [FDR control](fdr_control.md) | BH, BY, WCS, derandomized e-values, post-hoc FDP certificates, repeated batches, and online FDR distinctions |
| [Derandomized e-values](fdr_control.md#derandomized-conformal-e-values) | Repeated split-conformal evidence aggregated with e-values and e-BH |
| [Exchangeability martingales](exchangeability_martingales.md) | Sequential randomized ranks, betting martingales, alarms, and Ville scope |

## Evaluation and operations

| Page | Purpose |
|---|---|
| [Batch evaluation](batch_evaluation.md) | Reproducible labeled families with oddball and correct FDP/power bookkeeping |
| [Streaming evaluation](streaming_evaluation.md) | False-alarm, detection-delay, online-testing, and window-family designs |
| [Input validation](input_validation.md) | Enforced constraints, calibration resolution, fitted state, and weighted batch identity |
| [Best practices](best_practices.md) | Leakage prevention, family and episode design, reproducibility, and production review |
| [Logging](logging.md) | Actual logger namespaces and progress controls |
| [Troubleshooting](troubleshooting.md) | Symptom-led diagnosis without weakening guarantees post hoc |

!!! important "Guarantees are conditional statements"

    Standard split-conformal p-values require exchangeability of calibration
    and true-null test scores conditional on a fixed training-only scorer. FDR
    control additionally requires the dependence assumptions of the selection
    procedure. Weighted workflows require a correct shift model, overlap, and
    suitable weights. Sequential Ville guarantees require conditionally valid
    sequential p-values and a valid e-process. Passing API validation does not
    establish any of these assumptions.

## Review checklist

Before relying on a result, confirm:

- the null population, testing family, or monitoring episode was defined in
  advance;
- fitting, calibration, tuning, and final evaluation roles are separated;
- learned preprocessing is part of the fitted scoring construction;
- detector score polarity and fixed-state behavior are verified;
- empirical p-value resolution is adequate for the actual family;
- the selected FDR, FDP-bound, weighted, or sequential theorem matches the
  implementation and data design;
- labeled evaluation data did not choose the result later reported on it; and
- assumptions and failure modes accompany every statistical claim.

For agents and advanced users, the [API reference](../api/index.md) exposes
signatures and public docstrings, while [Common workflows](../api/common_workflows.md)
provides independently runnable examples.
