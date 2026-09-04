---
description: "Evaluate per-observation anomaly testing and sequential exchangeability monitoring with the correct families, episodes, and metrics."
---

# Streaming evaluation

“Streaming anomaly detection” can refer to different statistical tasks. Choose
the task before choosing metrics or code.

| Task | Output | Appropriate control target |
|---|---|---|
| Test distinct hypotheses as they arrive | Per-observation rejections | Online FDR or another online multiple-testing criterion |
| Detect a distributional change in one ordered process | Alarm time | False-alarm probability and detection delay |
| Review a completed window | Batch discovery mask | FDR within the declared window family |

`nonconform` provides the second lane through sequential randomized ranks and
exchangeability martingales. The optional `online_fdr` dependency implements
separate online multiple-testing procedures. They are not interchangeable.

## Labeled stream generation

The optional `oddball` package can generate a reproducible labeled sequence:

```bash
pip install "nonconform[data]"
```

`OnlineGenerator` requires a loader callback that returns a DataFrame with a
`Class` column. `oddball.load` returns arrays by default, so pass
`as_dataframe=True`.

```python
from oddball import Dataset, OnlineGenerator, load

generator = OnlineGenerator(
    load_data_func=lambda: load(Dataset.WBC, as_dataframe=True),
    anomaly_proportion=0.1,
    n_instances=50,
    train_size=0.5,
    seed=42,
)

labels = [label for _, label in generator.generate()]
print("instances:", len(labels))
print("anomalies:", int(sum(labels)))
```

Across the full configured run, the generator places exactly
`int(n_instances * anomaly_proportion)` anomalies at randomized positions. It
samples with replacement from held-out normal and anomaly pools. The generated
sequence is therefore a controlled contamination benchmark, not a model of a
single persistent change point.

## End-to-end exchangeability monitoring

For a valid monitoring construction, separate normal-only proper training data
from normal-only priming data, fit the scorer once, and keep it frozen while the
stream is processed.

```python
from oddball import Dataset, OnlineGenerator, load
from sklearn.ensemble import IsolationForest

from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

generator = OnlineGenerator(
    load_data_func=lambda: load(Dataset.WBC, as_dataframe=True),
    anomaly_proportion=0.1,
    n_instances=50,
    train_size=0.7,
    seed=42,
)

x_normal_reference = generator.get_training_data().to_numpy()
split = len(x_normal_reference) // 2
x_fit = x_normal_reference[:split]
x_prime = x_normal_reference[split:]

monitor = ExchangeabilityMonitor(
    detector=IsolationForest(n_estimators=50, random_state=42),
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(ville_threshold=20.0)
    ),
    seed=42,
)
monitor.fit(x_fit).prime(x_prime)

records = []
for step, (x_row, label) in enumerate(generator.generate(), start=1):
    state = monitor.update(x_row.iloc[0])
    records.append(
        {
            "step": step,
            "label": int(label),
            "p_value": state.p_value,
            "martingale": state.martingale,
            "alarms": state.triggered_alarms,
        }
    )

first_alarm = next(
    (record for record in records if "ville" in record["alarms"]),
    None,
)
print("first Ville alarm:", first_alarm)
print("final martingale:", records[-1]["martingale"])
```

The labels are used only after monitoring to describe the generated stream.
Filtering or conditionally processing rows based on those labels would make the
online protocol unrealistic.

!!! warning "An alarm is not a pointwise anomaly label"

    A martingale alarm says that cumulative evidence crossed a configured
    threshold. Comparing `triggered_alarms` row by row with anomaly labels and
    calling the result precision or recall misstates the target. One anomalous
    observation can trigger later evidence, and a persistent change can be
    detected after several affected observations.

## Evaluate the monitoring target

A change-monitoring study needs separate null and changed episodes.

### Under no change

Report:

- probability of any alarm by each prespecified horizon;
- distribution of first-alarm times, with nonalarms treated as censored;
- number of independently restarted episodes and error-budget allocation;
- martingale, scorer, priming size, threshold, and all seeds.

For a valid nonnegative martingale starting at one, a Ville threshold
`1 / alpha` bounds the probability of ever crossing by `alpha` under the null.
An empirical simulation checks implementation and scenario behavior; it does
not prove the theorem's assumptions for deployment.

### Under a defined change

Report:

- detection probability by a prespecified post-change horizon;
- detection delay from the known change time;
- false alarms before the change;
- censored runs that never alarm; and
- performance across change magnitudes and change locations fixed before
  evaluation.

Do not report mean delay only among successful runs without also reporting the
detection probability.

### Runnable episode study

The following small simulation illustrates the bookkeeping. Its ten runs are
not enough for a precise false-alarm estimate.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, PowerMartingale
from nonconform.monitoring import ExchangeabilityMonitor

reference_rng = np.random.default_rng(42)
x_reference = reference_rng.normal(size=(500, 3))
fitted_detector = ConformalDetector(
    detector=IsolationForest(n_estimators=40, random_state=42),
    strategy=Split(n_calib=0.35),
    seed=42,
).fit(x_reference)

threshold = 20.0
change_step = 20
horizon = 40

def first_ville_alarm(stream: np.ndarray, seed: int) -> int | None:
    monitor = ExchangeabilityMonitor.from_split_detector(
        fitted_detector,
        martingale=PowerMartingale(
            epsilon=0.5,
            alarm_config=AlarmConfig(ville_threshold=threshold),
        ),
        seed=seed,
    )
    for step, state in enumerate(monitor.update_many(stream), start=1):
        if "ville" in state.triggered_alarms:
            return step
    return None

null_alarm_times = []
changed_alarm_times = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    null_stream = rng.normal(size=(horizon, 3))
    changed_stream = np.vstack(
        [
            rng.normal(size=(change_step, 3)),
            rng.normal(loc=4.0, size=(horizon - change_step, 3)),
        ]
    )
    null_alarm_times.append(first_ville_alarm(null_stream, seed))
    changed_alarm_times.append(first_ville_alarm(changed_stream, seed + 100))

false_alarm_rate = np.mean([time is not None for time in null_alarm_times])
pre_change_false_alarms = [
    time
    for time in changed_alarm_times
    if time is not None and time <= change_step
]
post_change_alarms = [
    time for time in changed_alarm_times if time is not None and time > change_step
]
detection_rate = len(post_change_alarms) / len(changed_alarm_times)
delays = [time - change_step for time in post_change_alarms]

print("false-alarm fraction by horizon:", false_alarm_rate)
print(
    "pre-change false-alarm fraction:",
    len(pre_change_false_alarms) / len(changed_alarm_times),
)
print("post-change detection fraction:", detection_rate)
print("observed post-change delays:", delays)
```

The scorer and calibration history are fixed across this conditional simulation,
while each monitored stream is newly generated. A broader end-to-end study can
also repeat training and calibration draws.

## Ville, restarted Ville, CUSUM, and Shiryaev-Roberts

`AlarmConfig` exposes four thresholds with different interpretations:

| Threshold | Statistic | Interpretation |
|---|---|---|
| `ville_threshold` | Product martingale | Ville probability-of-ever-crossing bound under a valid null martingale |
| `restarted_ville_threshold` | Harmonic restart-mixture e-process | Ville bound for the implemented restart mixture |
| `cusum_threshold` | CUSUM of log betting increments | Change-evidence trigger requiring separate calibration |
| `shiryaev_roberts_threshold` | Shiryaev-Roberts statistic | Change-evidence trigger requiring separate calibration |

If action is taken when either of two Ville-valid alarms fires, allocate error
across them. Enabling several alarms does not leave each one with the full
lifetime error budget.

## Online FDR is a separate workflow

Online FDR methods such as LORD test a sequence of distinct hypotheses and
adapt future test levels to past decisions. Their guarantees typically require
conditionally super-uniform null p-values relative to the past and stated
dependence conditions.

Repeated calls to `ConformalDetector.compute_p_value(...)` against one fixed
calibration ECDF are only marginally valid by default and do not automatically
satisfy that sequential condition. If online multiple testing is your target,
construct an appropriate p-value process and verify the exact theorem before
passing it to `online_fdr`.

Conversely, a conformal martingale detects cumulative evidence against
exchangeability. It does not label each incoming hypothesis under an online FDR
criterion.

## Windowed batch analysis

You may intentionally define each completed time window as a batch family and
call `select(window, alpha=...)`. Under the relevant assumptions, that targets
FDR within each window. It does not provide pooled FDR across all windows or an
anytime false-alarm guarantee.

If all observations form one retrospective family, collect the complete set of
unweighted p-values and apply one procedure after the horizon closes. That
delays decisions and is no longer an online workflow.

## Retraining and resets

`ExchangeabilityMonitor.reset()` clears rank history and evidence while keeping
the fitted scorer. Prime the new episode again before monitoring. Refitting the
scorer also starts a new episode.

Repeated episodes need explicit lifetime error accounting. A threshold that
controls false alarms for one episode does not automatically control the chance
of any false alarm over unlimited resets.

Do not retrain inside an episode and continue the same martingale as if nothing
changed. The score sequence would no longer be exchangeable conditional on one
fixed scoring rule.

## Performance measurement

Measure separately:

- detector scoring latency per row;
- sequential rank update time;
- martingale update time;
- end-to-end latency including serialization and feature computation; and
- memory growth with rank history.

The exact sequential conformalizer stores all scores in a sorted list. Rank
queries are logarithmic, while insertion is linear in current history length;
there is no sliding window. Benchmark at the episode lengths expected in
deployment.

## Reproducibility notes

- Set seeds for the data generator, detector, sequential conformalizer, and
  randomized evaluation protocol.
- Reconstruct an `OnlineGenerator` to reproduce its entire initial split and
  stream. `reset()` resets RNG after construction and is not the clearest
  reproduction of the first run.
- Record censored episodes and the exact stopping rule.
- Never choose a threshold from the same episodes used for final performance
  reporting.

For implementation details and alarm-state fields, see
[Exchangeability martingales](exchangeability_martingales.md).
