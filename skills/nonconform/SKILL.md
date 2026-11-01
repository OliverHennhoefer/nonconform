---
name: nonconform
description: Turn anomaly scores into decisions with a stated error rate using nonconform, a Python library for conformal anomaly detection. Use this skill whenever someone wants to flag anomalies, outliers, novelties, fraud, defects, intrusions or out-of-distribution points and the number of false alarms matters, for example when choosing an anomaly threshold, replacing a contamination rate or top-k cutoff, controlling the false discovery rate (FDR) or false positive rate, getting p-values for anomaly scores, calibrating scores from PyOD, scikit-learn, an LLM judge or embedding distances, or detecting when a data stream changed with a bounded false alarm probability. Also use it whenever code imports nonconform.
license: BSD-3-Clause
---

# nonconform

nonconform wraps an anomaly detector (scikit-learn, PyOD or your own) and turns its scores into conformal p-values by ranking them against scores of known-normal reference data. From those p-values it makes decisions with an error rate the user chooses:

- a flagged set from a batch, with a controlled **false discovery rate** (FDR);
- a score threshold for production, with a certified **false positive rate** (FPR) on normal data;
- an alarm on a stream, with a bounded probability of **ever** firing when nothing changed.

None of this needs a model of what anomalies look like. It needs normal reference data from the same source as the new data.

Why bother: a detector's own cutoff (PyOD's `contamination`, a top-k, a z-score rule) is a guess about how many anomalies exist. When the guess is wrong, every flag shifts and nobody can say how many flags are false. Calibration replaces the guess with a number the user picked.

## Install

```bash
pip install nonconform            # core: scikit-learn detectors, all decision methods
pip install "nonconform[pyod]"    # adds PyOD's detectors
```

Other extras: `[fdr]` for online FDR with `online-fdr`, `[probabilistic]` for KDE-based p-values, `[data]` for benchmark datasets. Python 3.12 or newer.

## Step 1: Pin down the decision

Work out which error rate the user cares about, because it decides the API. If the request is ambiguous, ask; the options are different promises.

| The user wants | Error rate | Use |
| --- | --- | --- |
| Which points in this batch to act on, with few false alarms among them | FDR: expected share of false alarms among flagged points | `detector.select(x_new, alpha=...)` |
| A fixed threshold to deploy, rarely firing on normal traffic | FPR: chance that a normal point is flagged | `detector.fpr_bounds(confidence=...)`, then `threshold_for(target_fpr)` |
| Whether and when a stream changed | Probability of any false alarm, ever | `ExchangeabilityMonitor` with a martingale |
| A bound on false alarms in this exact batch, at several cutoffs | Realized false discovery proportion, with confidence | `detector.fdp_bounds(x_new, confidence=...)` |

Translate the user's words: "at most 10% of what we send to the fraud team should be false" is FDR with `alpha=0.10`. "Flag no more than 1% of normal sessions" is FPR with `target_fpr=0.01`. "Tell me when the pump starts behaving differently, and don't cry wolf" is stream monitoring with a Ville threshold of `1 / alpha`.

## Step 2: Get clean reference data

Everything rests on a reference set of normal observations that looks like the normal part of the new data.

- **Fit and calibrate on reference data only.** Never fit on the batch you are judging; that puts its anomalies into the calibration set and hides them.
- **Remove known anomalies from the reference data.** A few unknown ones mostly make results conservative (fewer discoveries); many make them unreliable.
- **Use enough calibration points.** With `n` calibration scores, the smallest possible p-value is `1 / (n + 1)`. For BH over `m` test points at level `alpha`, a single isolated discovery needs `n + 1 >= m / alpha`: 500 test points at `alpha=0.05` need about 10,000 calibration points. Batches with many real anomalies need fewer, because BH's cutoffs grow with rank. For FPR certificates, 1,000 calibration scores at 95% confidence give a floor of about 3.85%; for lower targets, use more scores or conditional p-values (`references/workflows.md`, section 2).
- **Match the time window.** If normal behavior drifts, calibrate on the most recent normal period.

## Step 3: Write the code

Batch FDR, the most common case:

```python
import numpy as np
from pyod.models.iforest import IForest  # any PyOD or scikit-learn detector works

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(0)
x_reference = rng.normal(size=(5_000, 4))  # known-normal data
x_new = np.vstack([rng.normal(size=(480, 4)), rng.normal(5.0, 1.0, size=(20, 4))])

detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=0.5),  # share (or count) of reference rows held out for calibration
    seed=0,
)
detector.fit(x_reference)  # trains on one part, calibrates on the held-out part
flagged = detector.select(x_new, alpha=0.05)  # boolean mask; FDR controlled at 5%

print(f"flagged {flagged.sum()} of {len(x_new)}:", np.flatnonzero(flagged))
```

`select()` computes conformal p-values and applies Benjamini–Hochberg (BH). Pass `method="storey"` for adaptive BH when anomalies are common. `detector.last_result.p_values` holds the p-values from the latest call.

For everything else, `references/workflows.md` has a complete, tested snippet per task. Read the section you need:

- **Certified threshold (FPR)** from a detector, or from precomputed scores, including targets below the certificate's floor
- **Precomputed scores** (an LLM judge, embedding distances, any model's output): p-values and BH without wrapping a detector
- **Stream monitoring** with `ExchangeabilityMonitor`
- **Pre-fitted detector** with a separate calibration set (`calibrate()`)
- **Covariate shift** (weighted p-values), **groups** such as sites or sensors (Mondrian), **e-values** (`DerandomizedSplits`) and **FDP bounds**

## Step 4: Check the assumptions before trusting the output

The guarantees hold when these do. Check each one and tell the user about any that may fail.

- **Exchangeability.** Normal test points must be distributed like the calibration points. Drift, seasonality or a changed pipeline breaks this. For a known covariate shift, use weighted mode; for a stream, monitor instead of batch-testing.
- **A rule fixed in advance.** Choose `alpha`, the detector and its settings before looking at test results. Re-running until the flagged set looks right voids the guarantee.
- **The right strategy.** `Split` and detached calibration carry the finite-sample guarantees. `CrossValidation` and `JackknifeBootstrap` use data more efficiently but carry no finite-sample guarantee in nonconform; say so if you use them.
- **One testing family.** `select()` controls FDR over the batch you pass. Splitting a batch into ten calls makes ten families with ten separate guarantees.
- **Streams are different.** Repeated batch p-values are not sequential p-values. Use `ExchangeabilityMonitor`, and don't refit the scorer mid-stream.

nonconform refuses combinations it cannot back with a guarantee (for example Storey's method with weights) and says why in the error. Read the message instead of working around it.

## Step 5: Report the result with the right claim

State what was controlled, at what level, under which assumption, and what it does not promise. For example:

> Flagged 23 of 2,000 transactions at a 10% false discovery rate. On average, at most 10% of transactions flagged this way are false alarms, provided today's legitimate transactions resemble the 5,000 reference transactions. The rate is an average over repeated batches, not a promise for this one, and it says nothing about anomalies that were not flagged.

Avoid these claims, which are wrong:

- "Each flagged point has a 95% chance of being an anomaly." FDR describes the flagged set on average, not each point.
- "There are no other anomalies." The guarantee limits false alarms; it does not bound misses.
- Calling every point with `p < 0.05` "significant" without BH. With 1,000 normal points, about 50 would pass.

`references/guarantees.md` explains each guarantee, its fine print and how to phrase it. Read it before writing up results for a stakeholder.

## Detectors

- **PyOD**, including models picked by PyOD 3's ADEngine or od-expert skill: pass the unfitted model; nonconform fits copies itself. If ADEngine already chose a detector, keep the choice and replace its contamination cutoff with `select()` or an FPR certificate.
- **scikit-learn**: `IsolationForest`, `OneClassSVM`, `SGDOneClassSVM`, `LocalOutlierFactor(novelty=True)` and `EllipticEnvelope` are recognized, and their score direction is handled.
- **Custom detectors** need `fit`, `decision_function`, `get_params` and `set_params`, and must be copyable. Each row's score must not depend on the other rows in the batch. Pass `score_polarity="higher_is_anomalous"` or `"higher_is_normal"` to say which way the scores point; omitted, it defaults to higher-is-anomalous.

nonconform adjusts some detector parameters it finds in `get_params()`: it sets `contamination` to a tiny value (the conformal rule replaces it), parallelism parameters such as `n_jobs` to `-1`, and `random_state` (or `seed`) from the `seed` you pass. Mention this if the user expects their own values to be used.

## Troubleshooting

- **`select()` flags nothing and logs a p-value floor warning**: the calibration set is too small for the batch size and `alpha`. See Step 2.
- **Almost everything is flagged**: the reference data do not match the new normal data (drift, a different pipeline, unscaled features), or a custom detector's score direction is inverted.
- **`threshold_for()` returns `inf` and `certificate.select()` flags nothing**: the target is below the certificate's floor. Use more calibration scores or conditional p-values (`references/workflows.md`, section 2).
- **Results change between runs**: pass `seed=` to `ConformalDetector`; it also seeds the base detector's `random_state`.
- **Progress bars and INFO lines clutter the output**: nonconform logs at INFO by default. Call `logging.getLogger("nonconform").setLevel(logging.WARNING)` before fitting; warnings still show.
