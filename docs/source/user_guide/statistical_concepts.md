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

**What it is**: A number in $[0,1]$ whose null distribution is super-uniform
when the test's assumptions hold. Smaller values represent stronger evidence
against that null; a p-value is not an effect size or a posterior probability.

**In nonconform**: A conformal p-value compares a test score with calibration
scores. Smaller values mean stronger evidence that the point does not look like
the calibration data.

**Guarantee**: For $a\in[0,1]$, a valid null p-value satisfies
$\Pr(p \le a) \le a$. For split conformal anomaly testing, this is ordinarily
a marginal statement over the random calibration set and a null test point,
conditional on an independently fitted, fixed scoring rule.

**Common mistake**: A small p-value is not a probability that the point is an
anomaly. It is evidence against the point behaving like the calibration
reference population.

**Classical vs. Randomized**:

- `Empirical()` defaults to `tie_break="classical"`, which gives discrete
  p-values in steps of $1/(n+1)$ and includes calibration scores tied with the
  test score, giving deterministic conservative behavior.
- Valid `tie_break` values are `"classical"` and `"randomized"` (or
  `TieBreakMode.CLASSICAL` / `TieBreakMode.RANDOMIZED`); `None` is invalid.
- For smoother p-values, use `Empirical(tie_break="randomized")`.
- `Probabilistic()` returns a continuous KDE estimate of score-tail
  probability. It is model-based and does not inherit the empirical rank's
  exact finite-sample validity. The library does not attach a general
  asymptotic-validity claim to it.

---

## E-values

**What it is**: A non-negative evidence value where larger values indicate
stronger evidence against a null hypothesis. Unlike p-values, e-values are
designed to be averaged across certain dependent analyses when their validity
conditions hold.

**In nonconform**: `select_conformal_e_values(...)` validates repeated
split-conformal result snapshots, builds uniformly aggregated conformal
evidence, and applies e-BH for batch FDR control. The lower-level
`conformal_e_values(...)` interface is available when experts have independently
verified score provenance.

**Guarantee**: The construction targets an aggregate null-evidence condition
that is sufficient for e-BH FDR control under the method assumptions; it does
not claim that each constructed value is individually an ordinary e-value. The
scope requires exchangeable inliers/null test points, a strict or independently
randomized score ordering, one fixed test family in consistent row order, valid
integrated unweighted `Split` score maps, and one final e-BH filtering step.

**Common mistake**: Do not treat e-values as p-values or threshold them at
ordinary p-value cutoffs. Use `e_value_false_discovery_control(...)` or
`select_conformal_e_values(...)` for FDR decisions.

---

## False Discovery Rate (FDR)

**What it is**: Let $V$ be the number of false discoveries and $R$ the total
number of discoveries in one testing family. Its realized false discovery
proportion is $\mathrm{FDP}=V/\max(R,1)$. The false discovery rate is the
repeated-sampling expectation $\mathrm{FDR}=\mathbb{E}[\mathrm{FDP}]$.

**Why it matters**: When you test many observations, some normal points will
look anomalous by chance. FDR control targets the expected false-positive
proportion among discoveries, for example at most 5% in expectation when the
assumptions hold.

**In nonconform**: `detector.select(X_test, alpha=...)` applies BH in standard
mode and WCS in weighted mode. `alpha` is a nominal FDR target, not a bound on
the FDP of every realized family. The corresponding guarantee requires valid
p-values and the assumptions of the selected procedure. The historical
`false_discovery_rate(...)` metric returns realized FDP for supplied labels.

---

## Anytime False-Alarm Control (Ville Bound)

**What it is**: A sequential false-alarm guarantee for nonnegative evidence
processes. If $M_t$ is a valid nonnegative supermartingale under the null and
$M_0\le1$, then for any $\lambda>0$:

$$
\Pr\left(\sup_t M_t \ge \lambda\right) \le \frac{1}{\lambda}.
$$

**In nonconform**: `ExchangeabilityMonitor` supplies randomized sequential-rank
p-values to the configured martingale. `AlarmConfig(ville_threshold=lambda)`
uses this style of anytime alarm thresholding for the product exchangeability
martingale.
`AlarmConfig(restarted_ville_threshold=lambda)` applies the same Ville threshold
to a restarted mixture e-process (evidence process) designed for sensitivity to
changes that begin later in the monitored stream. This is not a uniform power
improvement because later restart times receive less prior mass. The restart
prior is the
weighting over possible restart times; see
[Exchangeability Martingales](exchangeability_martingales.md#interpreting-restarted_ville_threshold)
for the documented default.

This guarantee applies to false alarms over time on a single stream. For
multiple testing settings across many hypotheses or streams, use dedicated FDR
procedures; see [Exchangeability Martingales](exchangeability_martingales.md)
and [FDR Control](fdr_control.md).

Repeatedly calling `ConformalDetector.compute_p_value()` uses one fixed
calibration ECDF and does not by itself supply the conditional sequential
validity needed here. Enabling both ordinary and restarted Ville alarms also
requires error allocation if action is taken when either alarm fires.

---

## Exchangeability

**What it is**: Data points are exchangeable if shuffling their order does not
change their joint distribution. For many practical workflows, "same population,
same measurement process, no systematic time/order effect" is the operational
check.

**Why it matters**: With a scoring rule fitted independently and then frozen,
exchangeability of calibration scores and the relevant true-null test score
supports the standard split-conformal rank argument.

**Operational evidence**: Calibration and target null observations are sampled
under the same mechanism, preprocessing and inclusion rules do not depend on
their eventual scores, and there is no unmodeled ordering or group effect. These
checks support exchangeability but cannot prove it from one dataset.

**Common violations**: Distribution shift, temporal dependence, duplicated or
clustered observations treated as independent rows, outcome-dependent sampling,
and preprocessing fitted using calibration or target information.

---

## Calibration Set

**What it is**: Reference observations scored by a fixed detector to form the
empirical null comparison set. In `Split`, these are held out from detector
fitting; detached calibration accepts a separately supplied set.

**Why it matters**: The calibration set provides the "baseline" for computing p-values. Test scores are compared against calibration scores.

**How big should it be**: Empirical conformal p-values move in steps of
$1/(n+1)$ with `n` calibration samples. Small calibration sets give coarse
p-values and can make FDR selection conservative or powerless. There is no
universal minimum: choose the count from the required testing resolution,
family size, detector fitting needs, and measured stability.

---

## Statistical Power

**What it is**: The probability of rejection under a specified alternative.
Across a realized labeled family, the analogous descriptive quantity is recall
or true positive rate, $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$.

**In nonconform**: `statistical_power(y_true, predictions)` retains its v1 name
but computes the realized true positive rate for those arrays. Repeated
simulation or sampling under a declared alternative is needed to estimate
power.

**Trade-off**: For a fixed scoring system and data design, relaxing the
rejection threshold typically increases both power and false positives. Choose
the FDR level from the operational cost of false alarms versus missed
anomalies.

---

## Covariate Shift

**What it is**: A shift from a calibration covariate distribution $P_X$ to a
target distribution $Q_X$ while the relevant conditional data-generating
mechanism remains invariant. Weighted conformal methods use the density ratio
$w(x)=dQ_X/dP_X$ and require target support to be covered by calibration
support.

**Solution**: Use weighted conformal prediction only when the shift is plausibly
covariate shift with sufficient support overlap and reliable weights. If the
anomaly mechanism changes, weighting alone does not restore the guarantees. See
[Weighted Conformal](weighted_conformal.md).

---

## Key Relationships

| Concept | Statistical role | Depends on |
|---------|----------|-------------|
| **p-value** | Supports a level-$a$ test through null super-uniformity | Null and calibration assumptions; calibration size sets rank resolution |
| **FDR** | Expected FDP of a declared testing family | Null p-value validity, dependence, family definition, and selection procedure |
| **Ville threshold** | Bounds ever-crossing probability for one valid stream | e-process validity and threshold choice |
| **Restarted Ville threshold** | Applies the Ville bound to a restart-mixture e-process | Component e-process validity and restart prior |
| **Power** | Rejection probability under an alternative | Alternative distribution, scorer, calibration, and decision rule |
| **Exchangeability** | Supplies rank symmetry used for conformal validity | Joint sampling, dependence, ordering, and preprocessing |

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
