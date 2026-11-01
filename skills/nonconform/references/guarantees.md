# Guarantees and how to state them

Read this before writing up results for a stakeholder, or when a user asks what a number means. Each section gives the claim, what it needs, what it does not promise, and a sentence that states it correctly.

Contents:

1. What every guarantee assumes
2. Conformal p-values
3. FDR from `select()`
4. Weighted selection (WCS)
5. E-values from `DerandomizedSplits`
6. FPR certificates for thresholds
7. FDP bounds for one batch
8. Stream monitoring
9. What no guarantee covers
10. Checking the assumptions in practice

## 1. What every guarantee assumes

- **A fixed score map.** The detector is trained on data separate from the calibration and test data (`Split` does this), learned preprocessing such as scaling sits inside the detector pipeline, and nothing is tuned on calibration or test results.
- **Reference data like the new normal data.** Calibration inliers and the normal points being tested come from the same population and measurement process, with no time trend or ordering effect. Formally: exchangeable. FPR certificates need more: independent draws from one distribution (i.i.d.).
- **A rule fixed in advance.** `alpha`, the method, the detector and its settings are chosen before seeing the test results. Trying several and reporting the best one voids the guarantee.
- **One family per guarantee.** A guarantee covers the batch or stream it was computed on.

If any of these fails, say so next to the result instead of reporting the guarantee.

## 2. Conformal p-values

**Claim.** For a normal test point, `P(p <= a) <= a` for every level `a`. The probability is over the calibration draw and the test point, so it is an average over calibration sets, not a statement about this one.

**Needs.** Section 1. Classical p-values take the values `k / (n + 1)` for `n` calibration scores, so the smallest is `1 / (n + 1)`.

**Does not promise.** A small p-value is not the probability that a point is normal or anomalous. It says how unusual the score is compared with the reference data.

**Say it like this.** "This transaction scored higher than all but 3 of the 2,500 reference transactions (p = 0.0016)."

## 3. FDR from `select()`

**Claim.** `select(x, alpha=a)` applies Benjamini–Hochberg (BH) to conformal p-values. The expected false discovery proportion among flagged points is at most `a` (in fact at most `a` times the share of normal points). The expectation is over repeated draws of calibration and test data (Bates et al., 2023).

**Needs.** Section 1, with `Split` or detached calibration. `method="storey"` (adaptive BH) has the same guarantee with classical `Empirical` p-values and continuous scores; choose it before looking at results. With `groups=`, p-values are computed within each group, and BH runs once over the whole batch.

**Does not promise.**

- Anything about a single batch. In a given batch, the share of false alarms can exceed `a`. For a high-confidence bound on this batch, use `fdp_bounds()` (section 7).
- Anything per point: a flagged point is not "90% likely" to be an anomaly at `alpha=0.1`.
- Anything about misses. Unflagged anomalies are not bounded.
- Control across batches. Running `select()` daily controls FDR within each day, not over the pooled discoveries of a month.
- A finite-sample guarantee with `CrossValidation` or `JackknifeBootstrap`. They are accepted, but the theory does not cover them. Say so.

**Say it like this.** "We flagged 23 of 2,000 transactions at a 10% false discovery rate. On average over batches like this one, at most 10% of transactions flagged this way are legitimate, provided today's legitimate transactions look like the 5,000 reference transactions. It is not a promise about this batch, and it says nothing about fraud that was not flagged."

## 4. Weighted selection (WCS)

**Claim.** With a `weight_estimator`, `select()` runs weighted conformalized selection (Jin & Candès, 2023). It controls FDR at `alpha` under covariate shift when the weights are the true density ratios between new and reference features. In practice the weights are estimated, so the control is approximate.

**Needs.** A real covariate shift: the feature mix changed, but what counts as normal did not. The new data's features must lie where the reference data has support. The weights depend on the whole batch, so do not split it into chunks.

**Estimated and clipped weights.** Errors in the estimated density ratios carry over into the FDR, and no stated bound covers them. Clipping (`clip_quantile=0.05`, the 5th to 95th percentile of all weights, by default) replaces the estimated ratio with a truncated one: it trades bias for stability and does not prove overlap. Report the estimator and the clipping level, and check how much the selection changes with the clipping (section 6 of `workflows.md`).

**Does not promise.** Exact FDR control with estimated or clipped weights, or protection against a changed anomaly mechanism, new measurement semantics or missing support.

**Calibration size.** Weighted mode often needs more calibration points than BH for the same batch. A test point's own weight sets its smallest possible p-value, and points far from the reference data, which is where anomalies sit, tend to get large weights. An empty weighted selection logs no p-value floor warning, so check the size yourself.

**Say it like this.** "Assuming only the feature mix shifted, the selection targets a 10% false discovery rate. That guarantee is exact only for the true density ratios. These weights were estimated by logistic regression and clipped to their 5th to 95th percentiles, so the 10% is approximate. After clipping, weights ranged from 0.14 to 3.17, and the effective calibration size was 1,145 of 2,000 points. Clipping at the 1st or 10th percentile instead flagged the same 5 points."

## 5. E-values from `DerandomizedSplits`

**Claim.** `DerandomizedSplits` averages evidence over several train/calibration splits and selects with e-BH, which controls FDR at `alpha` under its assumptions. It removes the dependence of the result on one random split.

**Needs.** Section 1, unweighted `Split` maps, one fixed batch, and one final e-BH step (which `select()` does).

**Does not promise.** More discoveries: it is usually less powerful than one large split. The individual values are not ordinary e-values or p-values; do not threshold them yourself.

## 6. FPR certificates for thresholds

**Claim.** `fpr_bounds(confidence=c)` (or `FPRCertificate.from_scores`) returns a curve that bounds the false positive rate of every raw-score threshold at once. With probability at least `c` over the calibration draw, the true FPR of each threshold is below its bound. `threshold_for(target)` inverts the curve, so with probability at least `c` the threshold flags at most `target` of future normal points. Because the bound holds at every threshold at once, choosing the threshold from the curve is safe. The statement is about the calibration set in hand, which is stronger than an average over calibration sets.

**Needs.** Clean calibration data with no anomalies, and calibration and future normal scores that are i.i.d. from the same distribution. Exchangeability alone is not enough. The scorer is fixed and was not tuned on the calibration data. Native certificates need unweighted `Split` or detached calibration.

**Floors.** Bounding every threshold at once costs tightness. At 95% confidence, the smallest target with a finite threshold is about:

| Calibration scores | 500 | 1,000 | 2,000 | 3,000 | 5,000 | 10,000 |
| --- | --- | --- | --- | --- | --- | --- |
| Smallest certifiable FPR | 5.5% | 3.9% | 2.7% | 2.2% | 1.7% | 1.2% |

Below the floor, `threshold_for()` returns infinity and `select()` flags nothing. For a lower target, gather more clean reference scores or use conditional p-values (below).

**Low targets with conditional p-values.** `ConditionalEmpirical` from `nonconform.scoring` makes p-values that are valid conditional on the calibration set: with probability at least `1 - delta` over the calibration draw, a future normal point has `P(p <= t) <= t` at every level `t` (Bates et al., 2023). Flagging `p <= target` therefore keeps the FPR at or below `target` with confidence `1 - delta`, under the same i.i.d. assumptions. The Simes map (`method="simes"`) is tighter than the certificate's band at small targets: at 95% confidence, 3,000 calibration scores certify a 1% target (the certificate's floor is 2.2%), and 1,000 scores certify 2%. Section 2 of `workflows.md` shows the code. Very small calibration sets still flag nothing: 500 scores cannot certify 1%.

**Does not promise.** FDR in a batch that contains anomalies, recall, or "no false alarm in a batch of 10,000": at a 1% FPR, a batch of 10,000 normal points still produces about 100 false alarms. It also does not bound the realized share of false alarms in any finite batch, only the rate.

**Say it like this.** "With 95% confidence, a conversation that is actually fine has at most a 5% chance of scoring 0.62 or higher. That figure comes from 3,000 conversations that reviewers marked fine, and it holds as long as future fine conversations look like them."

## 7. FDP bounds for one batch

**Claim.** `fdp_bounds(x, confidence=c)` gives, for this batch, an upper bound on the realized false discovery proportion among points with `p <= t`, at every cutoff `t` at once. With probability at least `c`, jointly over the data and the certificate's Monte Carlo draws, all bounds hold.

**Needs.** Section 1, unweighted empirical `Split` or detached calibration, and one construction. Rebuilding the certificate until a bound looks good voids it. `certificate.select(t)` takes a p-value cutoff, not a target FDP.

**Say it like this.** "With 95% confidence, at most 16% of the 41 flagged points, so at most 6 of them, are false alarms."

## 8. Stream monitoring

**Claim.** `ExchangeabilityMonitor` turns each new score into a sequential rank and feeds it to a martingale. If the stream stays exchangeable with the reference data, the probability that the martingale ever reaches `ville_threshold = 1 / alpha` is at most `alpha`, over the entire stream, however long it runs (Ville's inequality).

**Needs.** One ordered stream, a scorer that stays fixed during the episode, and the monitor's own rank process. Repeated `compute_p_value()` calls are not sequential p-values; never feed them to a martingale. With `K` streams, a threshold of `K / alpha` per stream bounds the chance of any false alarm across all of them by `alpha`. If you act on both the ordinary and the restarted Ville alarm, split `alpha` between them.

**Does not promise.**

- The time of the change. The alarm comes after the change started, often well after. Report "changed by step 313", not "changed at step 313".
- Anything for CUSUM or Shiryaev–Roberts alarms. Those thresholds need their own calibration and carry no Ville guarantee.
- Detection. A small change can go unnoticed for a long time.

**Say it like this.** "The alarm fired at reading 313. If the pump had kept behaving like the reference period, the chance of any alarm over the whole monitoring period was at most 1%. So the change happened at or before reading 313; the readings alone do not say exactly when it started."

After an alarm, investigate, then start a new episode with fresh reference data if the new behavior is the new normal.

## 9. What no guarantee covers

- **Misses.** Error control limits false alarms; it never bounds the anomalies that were not flagged. Report recall only from labeled evaluation data.
- **Cause.** A flag says the point is unusual compared with the reference data, not why.
- **Leaks.** Fitting preprocessing on all data, calibrating on the test batch, or tuning `alpha` until the result looks right silently voids every guarantee above.

## 10. Checking the assumptions in practice

Run these before trusting a result, and report what they show:

- **Held-out normal data.** Score recent known-normal points that were not used for fitting or calibration, and count the share of classical p-values at or below 0.1. It should be near 10%, but expect more spread than binomial noise. All the points share one calibration set, and conformal validity is an average over calibration sets. With `n` calibration and `m` held-out points, the standard deviation of the share is about `sqrt(0.1 * 0.9 * (1/n + 1/m))`: 1.1 percentage points for `n = 2,500` and `m = 1,000`, against 0.95 from binomial noise alone. To see the spread directly, repeat the check with a few different `seed` values. A share well above that range means the reference data no longer matches, usually because of drift. A smaller share does not threaten validity: tied scores and `ConditionalEmpirical` p-values are conservative by design.
- **Time order.** If the data have timestamps, check whether normal scores trend over time, for example by running `ExchangeabilityMonitor` over the reference period in time order. A trend breaks exchangeability.
- **Duplicates and clusters.** Remove duplicate rows, and don't let near-identical records (the same user or device) sit in both training and calibration data.
- **Data roles.** Confirm that no test row and no calibration row was used to fit the detector or its preprocessing.
- **Calibration size.** Compute the smallest p-value `1 / (n + 1)` and compare it with `alpha / m` for a batch of `m` points (section 3), or check the FPR floor (section 6).
