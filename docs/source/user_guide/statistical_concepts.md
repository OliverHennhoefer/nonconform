# Statistical Concepts Quick Reference

A quick reference for the key statistical terms used throughout nonconform. For detailed explanations and mathematical foundations, see [Understanding Conformal Inference](conformal_inference.md).

---

## P-values

**What it is**: A number between 0 and 1 indicating how "extreme" an observation is compared to a reference distribution.

**In nonconform**: The p-value tells you the probability of seeing an anomaly score at least as extreme as this observation, assuming it's normal. Lower p-values = more likely to be an anomaly.

**Example**: A p-value of 0.02 means only 2% of normal observations would have a score this extreme.

---

## False Discovery Rate (FDR)

**What it is**: The expected proportion of false positives among all points you flag as anomalies.

**Why it matters**: When you test many observations, some will look anomalous by chance. FDR control ensures that at most (say) 5% of your "discoveries" are actually false positives.

**In nonconform**: Use `scipy.stats.false_discovery_control(p_values, method='bh')` to apply Benjamini-Hochberg FDR control.

---

## Exchangeability

**What it is**: Data points are exchangeable if shuffling their order doesn't change their statistical properties.

**Why it matters**: This is the key assumption for conformal prediction guarantees. If your calibration and test data are exchangeable, the p-values are valid.

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

**Solution**: Use weighted conformal prediction to adjust for the distribution difference. See [Weighted Conformal](weighted_conformal.md).

---

## Key Relationships

| Concept | Controls | Affected by |
|---------|----------|-------------|
| **p-value** | False positive rate (per test) | Calibration set size, detector quality |
| **FDR** | False positives among discoveries | p-value validity, number of tests |
| **Power** | True positive rate | FDR threshold, detector quality |
| **Exchangeability** | p-value validity | Data collection process, distribution shift |

---

## References

For full mathematical foundations and proofs:

- [Understanding Conformal Inference](conformal_inference.md) – Complete theory guide
- [FDR Control](fdr_control.md) – Multiple testing in detail
- [Weighted Conformal](weighted_conformal.md) – Handling distribution shift
