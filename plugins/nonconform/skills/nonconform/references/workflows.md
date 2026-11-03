# nonconform workflows

Each block is complete: imports, synthetic data, the call, and output. Replace the synthetic arrays with the user's data. Every snippet was run against nonconform 2.0.1.

Contents:

1. Certified threshold (FPR) from a detector
2. Certified threshold from precomputed scores, including low targets
3. Precomputed scores: p-values and BH
4. Stream monitoring
5. Pre-fitted detector with a separate calibration set
6. Covariate shift: weighted p-values
7. Groups such as sites or sensors (Mondrian p-values)
8. Bounds on realized false discoveries (FDP)
9. Repeated splits with e-values
10. Inspecting p-values and scores

## 1. Certified threshold (FPR) from a detector

Use this when the deliverable is a threshold that will run in production and the requirement is "flag at most X% of normal points".

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(3_000, 4))
x_new = np.vstack([rng.normal(size=(95, 4)), rng.normal(5.0, 1.0, size=(5, 4))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=1_000),  # FPR certificates want at least ~1,000 calibration scores
    seed=0,
).fit(x_reference)

certificate = detector.fpr_bounds(confidence=0.95)
threshold = certificate.threshold_for(0.05)  # raw-score threshold for a 5% FPR target
scores = detector.score_samples(x_new)  # oriented so that higher means more anomalous
flagged = certificate.select(scores, target_fpr=0.05)

print("threshold:", threshold)
print("flagged:", np.flatnonzero(flagged))
```

With probability at least `confidence` over the calibration draw, the certified bound holds at every threshold at once, so picking the threshold from the curve is safe. The guarantee needs clean calibration data and future normal scores that are i.i.d. from the same distribution; exchangeability alone is not enough here. A target below the floor (about 3.85% for 1,000 scores at 95%) returns an infinite threshold that flags nothing. Keep the threshold's NumPy float64 type when comparing scores, or use `certificate.select(...)`.

## 2. Certified threshold from precomputed scores, including low targets

Use this when scores already exist, for example from an LLM judge or an embedding-distance model, and normal items have been scored by the same scorer.

```python
import numpy as np

from nonconform.fdr import FPRCertificate

rng = np.random.default_rng(0)
reviewed_ok_scores = rng.beta(2, 8, size=3_000)  # scores of items reviewed as normal
new_scores = np.r_[rng.beta(2, 8, size=480), rng.beta(8, 2, size=20)]

certificate = FPRCertificate.from_scores(reviewed_ok_scores, confidence=0.95)
threshold = certificate.threshold_for(0.05)
escalate = certificate.select(new_scores, target_fpr=0.05)

print(f"threshold {threshold:.3f}; escalate {escalate.sum()} of {new_scores.size}")
```

Scores must point upward: higher means more anomalous. Negate them if the scorer works the other way. The scorer must be fixed: it must not have been tuned or trained on the calibration items.

**Targets below the certificate's floor.** The certificate cannot certify less than about 2.2% with 3,000 scores at 95% confidence (see `guarantees.md`, section 6). Conditional p-values reach lower targets with the same kind of guarantee:

```python
import numpy as np

from nonconform.scoring import ConditionalEmpirical

rng = np.random.default_rng(0)
reviewed_ok_scores = rng.beta(2, 8, size=3_000)
new_scores = np.r_[rng.beta(2, 8, size=480), rng.beta(8, 2, size=20)]

target_fpr = 0.01
conditional = ConditionalEmpirical(method="simes", delta=0.05)  # delta = 1 - confidence
p_values = conditional.compute_p_values(new_scores, reviewed_ok_scores)
escalate = p_values <= target_fpr

# The same rule as a raw-score threshold: flag scores >= threshold.
grid = np.unique(reviewed_ok_scores)
grid_p = conditional.compute_p_values(grid, reviewed_ok_scores)
threshold = grid[grid_p <= target_fpr].min()  # raises if nothing is certifiable

print(f"threshold {threshold:.3f}; escalate {escalate.sum()} of {new_scores.size}")
```

With probability at least `1 - delta` over the calibration draw, the rule flags at most `target_fpr` of future normal items. The threshold rule flags a subset of what `p_values <= target_fpr` flags, so it keeps the guarantee. In a detector workflow, pass `estimation=ConditionalEmpirical(method="simes", delta=0.05)` to `ConformalDetector` and threshold `compute_p_values(x)` the same way.

## 3. Precomputed scores: p-values and BH

Use this for a batch decision with FDR control when you only have scores.

```python
import numpy as np
from scipy.stats import false_discovery_control

from nonconform.scoring import calculate_p_val

rng = np.random.default_rng(0)
calibration_scores = rng.normal(size=5_000)  # scores of known-normal items
test_scores = np.r_[rng.normal(size=480), rng.normal(5.0, 1.0, size=20)]

p_values = calculate_p_val(test_scores, calibration_scores)  # higher score = more anomalous
adjusted = false_discovery_control(p_values, method="bh")
flagged = adjusted <= 0.05  # BH at FDR 5%

print(f"flagged {flagged.sum()} of {test_scores.size}")
```

`calculate_p_val` returns classical conformal p-values, `(1 + #{calibration >= test}) / (n + 1)`. Ties, such as integer judge scores, make them conservative but keep them valid. With a detector instead of scores, prefer `ConformalDetector.select()`, which does all of this and records the provenance.

## 4. Stream monitoring

Use this to decide whether and when a stream stopped looking like the reference data, with a bounded false alarm probability over the whole stream.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(1_000, 3))
x_stream = np.vstack([rng.normal(size=(300, 3)), rng.normal(2.0, 1.0, size=(200, 3))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=0.5),
    seed=0,
).fit(x_reference)

alpha = 0.05
monitor = ExchangeabilityMonitor.from_split_detector(
    detector,
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(ville_threshold=1 / alpha)
    ),
    seed=0,
)
states = monitor.update_many(x_stream)  # or monitor.update(row) one observation at a time
first = next((i for i, s in enumerate(states) if s.triggered_alarms), None)

print("first alarm at stream index:", first)
```

The monitor primes its rank history with the calibration scores, then turns each new score into a sequential randomized rank and updates the martingale. If the stream stays exchangeable with the reference data, the probability that the martingale ever crosses `1 / alpha` is at most `alpha`. The alarm says the stream changed by then; the change itself may have started earlier. CUSUM and Shiryaev–Roberts thresholds in `AlarmConfig` are change triggers that need their own calibration and do not carry the Ville guarantee. Never feed `compute_p_value()` results into a martingale yourself, and never refit the detector during a monitoring episode.

## 5. Pre-fitted detector with a separate calibration set

Use this when the detector is already trained and a separate set of normal data is available for calibration.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(0)
x_train = rng.normal(size=(2_000, 4))
x_calibration = rng.normal(size=(2_000, 4))  # normal data the model never saw
x_new = np.vstack([rng.normal(size=(95, 4)), rng.normal(5.0, 1.0, size=(5, 4))])

base = IsolationForest(random_state=0).fit(x_train)
detector = ConformalDetector(detector=base, strategy=Split(n_calib=0.5), seed=0)
detector.calibrate(x_calibration)  # the whole array is the calibration set

flagged = detector.select(x_new, alpha=0.05)
print("flagged:", np.flatnonzero(flagged))
```

`n_calib` is ignored here but still required by `Split`. The model must stay fixed while calibration and test data are scored.

## 6. Covariate shift: weighted p-values

Use this when the new data's feature distribution has shifted from the reference data in a way you can model (the relationship between features and normality stays the same).

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(5_000, 3))
x_new = np.vstack([rng.normal(0.5, 1.0, size=(95, 3)), rng.normal(6.0, 1.0, size=(5, 3))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=0.4),  # 2,000 calibration points
    weight_estimator=logistic_weight_estimator(),
    seed=0,
).fit(x_reference)

flagged = detector.select(x_new, alpha=0.1)  # weighted conformalized selection (WCS), not BH
result = detector.last_result
w = result.calib_weights
print("flagged:", np.flatnonzero(flagged))
print("calibration weight range:", w.min().round(2), "to", w.max().round(2))
print("effective calibration size:", round(w.sum() ** 2 / (w**2).sum()), "of", w.size)

# Sensitivity to clipping: report it next to the result; the pre-chosen setting stays the answer.
for clip in (0.01, 0.10):
    alternative = ConformalDetector(
        detector=IsolationForest(random_state=0),
        strategy=Split(n_calib=0.4),
        weight_estimator=logistic_weight_estimator(clip_quantile=clip),
        seed=0,
    ).fit(x_reference)
    print(f"clip_quantile={clip}: flagged", np.flatnonzero(alternative.select(x_new, alpha=0.1)))
```

Validity rests on the covariate-shift model, overlapping support and reasonable density-ratio estimates. The FDR control is exact only for the true density ratios, so with estimated weights report it as approximate. Check the weights for extreme values, and report the effective calibration size, the clipping level and how the selection changes with it. Here, clipping at the 1st, 5th or 10th percentile flags the same 5 points. Without clipping (`clip_quantile=None`), nothing is flagged: the five far-out points lie outside the reference data's support, their estimated weights reach 1,500 to 20,000 against about 2,000 for all calibration points combined, and their p-values stay above 0.4. Weighted mode needs more calibration data than BH: with `Split(n_calib=500)`, this example flags nothing in weighted mode and logs no floor warning, while unweighted BH on the same 500 calibration points flags all five. Storey's method, groups and certificates do not combine with weights.

## 7. Groups such as sites or sensors (Mondrian p-values)

Use this when groups have different normal score levels and each test point should be compared only with reference points from its own group.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(0)
x_reference = np.vstack([rng.normal(0.0, 1.0, size=(1_500, 3)), rng.normal(0.0, 2.0, size=(1_500, 3))])
site_reference = np.repeat(["a", "b"], 1_500)
x_new = np.vstack([rng.normal(0.0, 1.0, size=(95, 3)), rng.normal(6.0, 1.0, size=(5, 3))])
site_new = np.repeat(["a"], 100)

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=0.5),
    seed=0,
).fit(x_reference, groups=site_reference)

flagged = detector.select(x_new, alpha=0.05, groups=site_new)
print("flagged:", np.flatnonzero(flagged))
```

P-values are valid within each group even when the test batch has a different group mix than the reference data. BH is applied once over the whole batch. Groups work with unweighted `Split`, detached `Split`, `CrossValidation` and `JackknifeBootstrap` and classical `Empirical` p-values; Storey, weights, certificates and e-values reject groups.

## 8. Bounds on realized false discoveries (FDP)

FDR is an average over repeated batches. Use this when the user wants a high-confidence bound on false discoveries in this specific batch, across several p-value cutoffs.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(3_000, 4))
x_batch = np.vstack([rng.normal(size=(160, 4)), rng.normal(5.0, 1.0, size=(40, 4))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=1_000),
    seed=0,
).fit(x_reference)

certificate = detector.fdp_bounds(x_batch, confidence=0.95)
print(certificate.to_frame(thresholds=[0.001, 0.01, 0.05]))
mask = certificate.select(0.01)  # p-value cutoff, not a target FDP
print("discoveries:", mask.sum(), "FDP upper bound:", certificate.bound_at(0.01))
```

The bounds hold simultaneously over all cutoffs with probability at least `confidence`, jointly over the data and the certificate's Monte Carlo sampling. Bounds vary slightly between constructions; build one certificate and do not rebuild until a bound looks good.

**Short lists from large batches.** The default envelope (`mc_thc`, truncated to the range 0.01 to 0.99) is loose at cutoffs far below 0.01, which is where a short list from a large batch sits. If you expect the list to be a small share of the batch, say under 5%, decide before computing any bound to use the expert route with `method="mc_hc"`:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import FDPCertificate

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(5_000, 4))
x_batch = np.vstack([rng.normal(size=(1_960, 4)), rng.normal(5.0, 1.0, size=(40, 4))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=3_500),
    seed=0,
).fit(x_reference)
flagged = detector.select(x_batch, alpha=0.1)
result = detector.last_result

# Envelope chosen before computing any bound: the list is expected to be short.
certificate = FDPCertificate.from_p_values(
    result.p_values,
    n_calibration=len(result.calib_scores),
    confidence=0.95,
    method="mc_hc",
)
cutoff = result.p_values[flagged].max()
print("flagged:", flagged.sum(), "FDP upper bound for this list:", certificate.bound_at(cutoff))
```

In simulations of this shape (2,000 test points, 3,500 calibration scores, 40 anomalies, BH lists of about 43 points), the default envelope bounded the FDP of the BH list at a median of 0.86 (0.72 to 1.00) over 40 runs, which says little. `mc_hc` gave a median of 0.23 (0.11 to 0.36) over 200 runs, and its bounds were never violated. The expert route leaves the provenance checks to you: use it only on p-values from unweighted `Split` or detached calibration with `Empirical` p-values, as here.

## 9. Repeated splits with e-values

`DerandomizedSplits` averages evidence over several train/calibration splits and applies e-BH, which removes the luck of a single split.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, DerandomizedSplits

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(1_000, 4))
x_new = np.vstack([rng.normal(size=(90, 4)), rng.normal(5.0, 1.0, size=(10, 4))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=DerandomizedSplits(n_repetitions=5, n_calib=0.3),
    seed=0,
).fit(x_reference)

flagged = detector.select(x_new, alpha=0.05)
evidence = detector.last_selection_result
print("flagged:", np.flatnonzero(flagged))
print("e-values of flagged points:", evidence.e_values[flagged].round(1))
print("e-value needed to be flagged:", evidence.e_threshold)
```

It is usually less powerful than one large split. After `select()`, diagnostics are in `last_selection_result`, and `last_result` is `None`.

## 10. Inspecting p-values and scores

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(1_000, 4))
x_new = np.vstack([rng.normal(size=(8, 4)), rng.normal(5.0, 1.0, size=(2, 4))])

detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=0.5),
    seed=0,
).fit(x_reference)

p_values = detector.compute_p_values(x_new)
result = detector.last_result
print(np.column_stack([result.test_scores, p_values]))
print("BH-adjusted:", result.adjusted_p_values(method="bh"))
```

`last_result` describes the most recent `score_samples`, `compute_p_value(s)` or `select` call. Thresholding `adjusted_p_values(...)` at `alpha` reproduces `select()`. For online FDR over an open-ended sequence of single tests, install `nonconform[fdr]` and follow the online FDR section of the [FDR guide](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/).
