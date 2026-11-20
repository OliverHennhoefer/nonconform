---
description: "Choose between a short list of anomalies with select() and a reusable false-alarm threshold with fpr_bounds(), and size the calibration set for either."
---

# Start here

Your detector gives every point a score. To act on the scores you need a rule
for which ones count as anomalies. nonconform offers two rules, and they make
different promises.

| You want | Call | Promise |
|---|---|---|
| **A short list** worth reviewing | `detector.select(x, alpha=0.1)` | On average, at most 10% of the flagged points are normal. |
| **A threshold** to replace a `contamination` guess | `detector.fpr_bounds()` | At most 5% (your target) of normal points get flagged. |

If your data arrive one point at a time and you want to know when the stream
changed, see [Monitor a sensor stream](use_cases/sensor_stream.md) instead.

## A short list: `select`

Fit on normal data only, then pass the new batch:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(5_000, 2))  # normal data only
x_batch = np.vstack([rng.normal(size=(200, 2)), rng.normal(loc=4.0, size=(20, 2))])

alpha = 0.1
detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=int(np.ceil(len(x_batch) / alpha))),  # 2,200
    score_polarity="auto",
    seed=42,
).fit(x_reference)

flagged = detector.select(x_batch, alpha=alpha)
print(f"Flagged {flagged.sum()} of {len(x_batch)}")
```

`alpha` is the share of false flags you accept among the flagged points,
averaged over batches. It is not a score threshold.

**The list can be empty.** That is the answer when the evidence cannot back the
promise, and it beats a list that is half wrong. There are two usual causes: the
calibration set is too small for the batch (next section), or the detector does
not separate the anomalies well. In the second case a threshold often still
works.

## How much calibration data

`Split(n_calib=...)` holds out reference points to calibrate the scores. Size it
from the batch:

```text
n_calib >= batch_size / alpha
```

The smallest p-value nonconform can give is `1 / (n_calib + 1)`. To flag a
single anomaly on its own, `select` needs a p-value at or below
`alpha / batch_size`. With fewer calibration scores it can still flag a group
of anomalies that all beat every calibration score, but a lone one goes
unflagged.

| Batch size | `alpha` | `n_calib` for one lone anomaly |
|---|---|---|
| 100 | 0.1 | 1,000 |
| 1,000 | 0.1 | 10,000 |
| 1,000 | 0.05 | 20,000 |

If the reference data are too small, score smaller batches or use a threshold.
When a selection comes back empty for this reason, nonconform logs the size it
would need. [Screen a batch for intrusions](use_cases/batch_screening.md) shows
the difference on real data.

## A threshold: `fpr_bounds`

Continuing from the detector above:

```python
certificate = detector.fpr_bounds(confidence=0.95)
scores = detector.score_samples(x_batch)
flagged = certificate.select(scores, target_fpr=0.05)
print(f"Threshold {certificate.threshold_for(0.05):.4f}, flagged {flagged.sum()}")
```

`contamination` asks you to guess the share of anomalies. `target_fpr` states the
share of normal points you can afford to flag, so the threshold does not depend
on how many anomalies arrive. You compute it once and apply it to every new
point. With 95% confidence over the calibration data, normal points cross it at
most 5% of the time.

A target needs enough calibration scores to have a finite threshold:

```text
n_calib >= ln(1 / (1 - confidence)) / (2 * target_fpr**2)
```

That is about 600 for a 5% target and 15,000 for a 1% target, both at 95%
confidence. Below that, `threshold_for` returns `inf` and nothing is flagged.
[Replace the contamination guess](use_cases/false_alarm_threshold.md) shows the
threshold on a dataset where `select` returns nothing.

## Which one?

- A person reviews what is flagged in each batch: use `select`.
- A rule runs unattended on every point and you budget false alarms: use
  `fpr_bounds`.
- You need both a short list and a cap on false alarms: they answer different
  questions, so pick the one your decision depends on.

Both rely on the reference data being clean and representative. Read
[Assumptions and guarantees](guarantees.md) before you report a number.
