# Statistical Concepts Quick Reference

A quick reference for the key statistical terms used throughout nonconform. For detailed explanations and mathematical foundations, see [Understanding Conformal Inference](conformal_inference.md).

---

## P-values

**What it is**: A number between 0 and 1 indicating how "extreme" an observation is compared to a reference distribution.

**In nonconform**: A conformal p-value measures how extreme an anomaly score is relative to calibration scores, assuming the point is normal and the relevant exchangeability or covariate-shift assumptions hold. Lower p-values = more evidence of anomaly.

**Example**: A p-value of 0.02 means the point is more extreme than almost all calibration examples. Under the null assumptions, conformal p-values are super-uniform: $\Pr(p \le 0.02) \le 0.02$.

**Classical vs. Randomized**:

- `Empirical()` defaults to `tie_break="classical"`, which gives discrete
  p-values in steps of $1/(n+1)$.
- Valid `tie_break` values are `"classical"` and `"randomized"` (or
  `TieBreakMode.CLASSICAL` / `TieBreakMode.RANDOMIZED`); `None` is invalid.
- For smoother (continuous) p-values, use
  `Empirical(tie_break="randomized")`.
- `Probabilistic()` also gives continuous p-values through KDE. It trades
  finite-sample guarantees for asymptotic guarantees (guarantees that become
  accurate as sample size grows).

---

## False Discovery Rate (FDR)

**What it is**: The expected proportion of false positives among all points you flag as anomalies.

**Why it matters**: When you test many observations, some will look anomalous by chance. FDR control targets the expected false-positive proportion among discoveries, for example at most 5% in expectation when the assumptions hold.

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

**What it is**: Data points are exchangeable if shuffling their order doesn't change their statistical properties.

**Why it matters**: This is the key assumption for standard conformal prediction guarantees. If your calibration and test data are exchangeable, split conformal p-values are marginally valid.

**When it holds**: Training and test data from the same source, collected the same way, without systematic changes over time.

**When it's violated**: Distribution shift, temporal drift, or different data collection procedures between training and test.

---

## Calibration Set

**What it is**: A held-out portion of training data used to compute reference anomaly scores.

**Why it matters**: The calibration set provides the "baseline" for computing p-values. Test scores are compared against calibration scores.

**How big should it be**: Generally 100+ samples for reliable p-values. Larger is better, but diminishing returns after ~1000.

---

## Statistical Power

**What it is**: The proportion of true anomalies that you successfully detect.

**In nonconform**: Use `statistical_power(y_true, predictions)` to measure this.

**Trade-off**: Higher power (detecting more anomalies) often means higher FDR (more false positives). Choose your FDR threshold based on the cost of false positives vs. missed anomalies.

---

## Covariate Shift

**What it is**: When the feature distribution P(X) differs between training and test data, but the relationship P(Y|X) stays the same.

**Example**: Training on data from Sensor A, testing on data from Sensor B (different readings, same underlying physics).

**Solution**: Use weighted conformal prediction only when the shift is plausibly covariate shift with sufficient support overlap and reliable weights. See [Weighted Conformal](weighted_conformal.md).

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

For full mathematical foundations and proofs:

- [Understanding Conformal Inference](conformal_inference.md) – Complete theory guide
- [FDR Control](fdr_control.md) – Multiple testing in detail
- [Weighted Conformal](weighted_conformal.md) – Covariate-shift workflows
