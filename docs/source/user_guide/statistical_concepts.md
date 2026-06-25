---
description: "Review statistical concepts behind nonconform, including p-values, FDR, exchangeability, calibration, and covariate shift."
---

# Statistical Concepts

A practitioner reference for the statistical terms used throughout nonconform.
Each entry says what the term means, how it shows up in the library, and what
can go wrong in practice.

For more detail, see [Understanding Conformal Inference](conformal_inference.md).

---

## P-values

**What it is**: A number between 0 and 1 that summarizes how extreme an
observation looks compared with a reference distribution.

**In nonconform**: A conformal p-value compares a test score with calibration
scores. Smaller values mean stronger evidence that the point does not look like
the calibration data.

**Guarantee**: Under the null assumptions, conformal p-values are
super-uniform: $\Pr(p \le a) \le a$. For example, a valid null p-value should
fall below 0.02 with probability at most 2%.

**Common mistake**: A small p-value is not a probability that the point is an
anomaly. It is evidence against the point behaving like the calibration
reference population.

**Classical vs. Randomized**:

- `Empirical()` defaults to `tie_break="classical"`, which gives discrete
  p-values in steps of $1/(n+1)$.
- Valid `tie_break` values are `"classical"` and `"randomized"` (or
  `TieBreakMode.CLASSICAL` / `TieBreakMode.RANDOMIZED`); `None` is invalid.
- For smoother p-values, use `Empirical(tie_break="randomized")`.
- `Probabilistic()` also gives continuous p-values through KDE. It trades
  exact finite-sample conformal validity for model-based/asymptotic behavior;
  validate it on your task before treating its p-values as calibrated.

---

## E-values

**What it is**: A non-negative evidence value where larger values indicate
stronger evidence against a null hypothesis. Unlike p-values, e-values are
designed to be averaged across certain dependent analyses when their validity
conditions hold.

**In nonconform**: `conformal_e_value_selection(...)` builds conformal e-values
from repeated split-conformal score arrays and applies e-BH for batch FDR
control. Use it when split randomness makes ordinary `Split` decisions unstable.

**Guarantee**: The derandomized conformal e-values target an average validity
condition that is sufficient for e-BH FDR control under the method assumptions:
exchangeable inliers/null test points, valid repeated split-conformal score
maps, and one final e-BH filtering step.

**Common mistake**: Do not treat e-values as p-values or threshold them at
ordinary p-value cutoffs. Use `e_value_false_discovery_control(...)` or
`conformal_e_value_selection(...)` for FDR decisions.

---

## False Discovery Rate (FDR)

**What it is**: The expected proportion of false positives among the points you
flag as anomalies.

**Why it matters**: When you test many observations, some normal points will
look anomalous by chance. FDR control targets the expected false-positive
proportion among discoveries, for example at most 5% in expectation when the
assumptions hold.

**In nonconform**: Prefer `detector.select(X_test, alpha=...)` for default FDR-controlled decisions. Use `scipy.stats.false_discovery_control(...)` when you intentionally need manual p-value post-processing.

---

## Anytime False-Alarm Control (Ville Bound)

**What it is**: A sequential false-alarm guarantee for martingale evidence
processes. If `M_t` is a valid nonnegative martingale under the null and starts
at 1, then for any threshold `lambda`:

$$
\Pr\left(\sup_t M_t \ge \lambda\right) \le \frac{1}{\lambda}.
$$

**In nonconform**: `AlarmConfig(ville_threshold=lambda)` uses this style of
anytime alarm thresholding for the product exchangeability martingale.
`AlarmConfig(restarted_ville_threshold=lambda)` applies the same Ville threshold
to a restarted mixture e-process (evidence process) with better sensitivity to
changes that begin later in the monitored stream. The restart prior is the
weighting over possible restart times; see
[Exchangeability Martingales](exchangeability_martingales.md#interpreting-restarted_ville_threshold)
for the documented default.

This guarantee applies to false alarms over time on a single stream. For
multiple testing settings across many hypotheses or streams, use dedicated FDR
procedures; see [Exchangeability Martingales](exchangeability_martingales.md)
and [FDR Control](fdr_control.md).

---

## Exchangeability

**What it is**: Data points are exchangeable if shuffling their order does not
change their joint distribution. For many practical workflows, "same population,
same measurement process, no systematic time/order effect" is the operational
check.

**Why it matters**: This is the key assumption for standard conformal prediction guarantees. If your calibration and test data are exchangeable, split conformal p-values are marginally valid.

**When it holds**: Training/calibration and test data come from the same source,
collected the same way, without systematic changes over time or sampling rules.

**When it's violated**: Distribution shift, temporal drift, or different data collection procedures between training and test.

---

## Calibration Set

**What it is**: A held-out portion of training data used to compute reference anomaly scores.

**Why it matters**: The calibration set provides the "baseline" for computing p-values. Test scores are compared against calibration scores.

**How big should it be**: Empirical conformal p-values move in steps of
$1/(n+1)$ with `n` calibration samples. Small calibration sets give coarse
p-values and can make FDR selection conservative or powerless. Treat 50-100
calibration samples as a usability floor, not a theorem; larger calibration
sets usually give better p-value resolution.

---

## Statistical Power

**What it is**: The proportion of true anomalies that you successfully detect.

**In nonconform**: Use `statistical_power(y_true, predictions)` to measure this.

**Trade-off**: Higher power often requires accepting more false positives.
Choose the FDR level from the operational cost of false alarms versus missed
anomalies.

---

## Covariate Shift

**What it is**: When the feature distribution P(X) differs between training and test data, but the relationship P(Y|X) stays the same.

**Example**: Training on data from Sensor A, testing on data from Sensor B (different readings, same underlying physics).

**Solution**: Use weighted conformal prediction only when the shift is plausibly
covariate shift with sufficient support overlap and reliable weights. If the
anomaly mechanism changes, weighting alone does not restore the guarantees. See
[Weighted Conformal](weighted_conformal.md).

---

## Key Relationships

| Concept | Controls | Affected by |
|---------|----------|-------------|
| **p-value** | Per-test false-positive probability under null assumptions | Calibration set size, detector quality |
| **FDR** | Expected false-positive proportion among discoveries | p-value validity, number of tests |
| **Ville threshold** | Anytime false alarm probability (per stream) | Martingale validity, threshold choice |
| **Restarted Ville threshold** | Anytime false alarm probability with better sensitivity to changes later in the stream | e-process validity, restart prior |
| **Power** | True positive rate | FDR threshold, detector quality |
| **Exchangeability** | p-value validity | Data collection process, distribution shift |

---

## References

For mathematical foundations and implementation context:

- [Understanding Conformal Inference](conformal_inference.md) - conformal
  p-values and exchangeability assumptions.
- [FDR Control](fdr_control.md) - multiple testing, BH selection, and dependence
  assumptions.
- [Weighted Conformal](weighted_conformal.md) - covariate-shift workflows.
- [Shafer & Vovk (2008)](https://jmlr.org/papers/v9/shafer08a.html) -
  conformal prediction tutorial.
- [Bates et al. (2023)](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-1/Testing-for-outliers-with-conformal-p-values/10.1214/22-AOS2244.short) -
  conformal p-values for outlier testing.
- [Benjamini & Hochberg (1995)](https://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf) -
  the original FDR procedure.
- [Jin & Candès (2023)](https://arxiv.org/abs/2307.09291) -
  weighted conformal p-values and WCS.
- [Ramdas et al. (2023)](https://arxiv.org/abs/2210.01948) - anytime-valid
  inference with e-values and martingales.
