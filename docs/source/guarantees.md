---
description: "What each nonconform workflow promises, the assumptions each promise needs, and what breaks them."
---

# Assumptions and guarantees

Every number nonconform reports is a promise that holds when its assumptions
hold. This page collects them in one place. Other pages link here rather than
repeat them.

## The assumption behind everything

**The calibration data are clean normal points, and new normal points look like
them.** Formally, calibration scores and the scores of new normal points are
exchangeable, given a detector fitted on separate data. `Split` keeps fitting
and calibration apart for you.

nonconform cannot check this. Passing input validation does not establish it.
If the reference data contain anomalies, or production data have drifted away
from them, the numbers below no longer mean what they say.

## What each workflow promises

| Workflow | Promise | Additionally needs | Does not promise |
|---|---|---|---|
| `select(x, alpha)` | Expected share of normal points among the flagged ones (FDR) is at most `alpha`. | The positive dependence that conformal p-values have under the assumption above. | The share in one particular batch, or finding every anomaly. |
| `select` with `DerandomizedSplits` | Same FDR promise, via e-BH. | The same batch in the same row order for every repetition, and one final e-BH step. | That each averaged value is an ordinary e-value. |
| `select` in weighted mode | FDR under covariate shift. | A correct shift model, overlap between calibration and test data, and reliable weights. | Validity when the weights are wrong. |
| `select` with `groups=` | FDR with each point ranked only within its group. | Exchangeability within each group, and groups fixed in advance. | Validity across group labels chosen after seeing results. |
| `fdp_bounds(x, confidence)` | With probability at least `confidence`, the realized share of false flags in this batch is below the bound at every cutoff at once. | A fixed batch and scoring map, and no re-running until the bound looks good. | Anything about other batches. |
| `fpr_bounds(confidence)` | With probability at least `confidence` over the calibration data, normal points cross the chosen threshold at most `target_fpr` of the time. | Calibration and future normal scores i.i.d. from one distribution. Exchangeability alone is not enough. Unweighted `Split` only. | FDR, the share in a finite batch, or recall. |
| `ExchangeabilityMonitor` with Ville threshold `c` | On one stream with no change, the chance of ever alarming is at most `1 / c`. | A detector fixed before monitoring starts. | FDR across many streams, or which point is anomalous. CUSUM and Shiryaev-Roberts thresholds need their own calibration. |

`Empirical()` p-values, the default, are exact in finite samples.
`Probabilistic()` p-values are a smooth model-based estimate and carry no
finite-sample guarantee.

## What breaks the guarantees

- Anomalies in the reference data. They inflate calibration scores and hide
  real anomalies.
- Drift between the reference period and the data you score.
- Wrong score polarity. If higher scores mean *normal* for your detector and
  nonconform is not told, every p-value is inverted. Set `score_polarity`, see
  [Detector compatibility](user_guide/detector_compatibility.md).
- Preprocessing fitted on calibration or test data. Fit it inside the detector,
  for example in a scikit-learn `Pipeline`.
- Choosing `alpha`, the detector or the seed after looking at the results on
  the data you report on.
- Treating one batch's realized share of false flags as the FDR.

## Read more

- [Statistical concepts](user_guide/statistical_concepts.md) defines p-values,
  FDR, FPR and exchangeability.
- [FDR control](user_guide/fdr_control.md) covers BH, e-BH, WCS and the FDP and
  FPR certificates in detail.
- [Weighted conformal](user_guide/weighted_conformal.md) covers covariate shift.
- [Exchangeability martingales](user_guide/exchangeability_martingales.md)
  covers sequential monitoring and alarm types.
