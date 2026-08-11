---
description: "Monitor exchangeability and distribution shift in streams using conformal martingales and nonconform utilities."
---

# Exchangeability Martingales

Monitor streaming conformal p-values for evidence against exchangeability.
Use these alarms as evidence signals, not as FDR-controlled anomaly decisions.

## What This Feature Does

`nonconform.monitoring` constructs randomized sequential conformal p-values
from a frozen anomaly scorer. `nonconform.martingales` consumes those p-values
and maintains:

- Martingale evidence (`M_n`)
- Restarted mixture e-process evidence for late-change sensitivity
- CUSUM statistic (cumulative-sum change evidence)
- Shiryaev-Roberts statistic (sequential evidence accumulator)
- Optional alarm triggers from configurable thresholds

Implemented methods in this release:

- `PowerMartingale`
- `SimpleMixtureMartingale`
- `SimpleJumperMartingale`

## Why P-values (Not Raw Scores)

These martingales are conformal/exchangeability tests. Their validity relies on:

- under exchangeability, the sequential conformal p-values used by the
  martingale are valid and, with proper randomized tie-breaking in the classical
  construction, i.i.d. `Uniform(0, 1)`

Raw anomaly scores do not satisfy this requirement directly. Neither does
repeatedly comparing stream observations with one fixed split-calibration ECDF:
those p-values are marginally valid but share calibration-induced dependence.
Use `ExchangeabilityMonitor` to apply randomized sequential ranks before the
martingale. The martingale classes do not repair invalid p-values, temporal
dependence, or detector retraining choices that break the conformal assumptions.

!!! warning "Do not mix up alarm types"
    `ville_threshold` and `restarted_ville_threshold` provide anytime
    false-alarm control for a single valid stream. They do not control FDR
    across many simultaneous hypotheses or many streams. For that, use the
    methods in [FDR Control](fdr_control.md).

## Rigorous Basic Usage

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_train = rng.standard_normal((300, 5))
x_reference = rng.standard_normal((100, 5))
x_stream = rng.standard_normal((100, 5))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.2),
    score_polarity="auto",
    seed=42,
)
detector.fit(x_train)

monitor = ExchangeabilityMonitor.from_split_detector(
    detector,
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(restarted_ville_threshold=100.0)
    ),
    seed=42,
)

for x_t in x_stream:
    state = monitor.update(x_t)
    if "restarted_ville" in state.triggered_alarms:
        print(
            "Restarted Ville alarm "
            f"at step={state.evidence_step}, "
            f"M={state.restarted_martingale:.2f}"
        )
        break
```

`from_split_detector(...)` copies the fitted scoring model and primes the
sequential rank history with its calibration scores. Martingale capital remains
at one until the first monitored observation. The original detector and its
fixed-calibration behavior are unchanged.

Alternatively, fit a monitor directly and supply a separate reference set:

```python
monitor = ExchangeabilityMonitor(
    IsolationForest(random_state=42),
    score_polarity="auto",
    seed=42,
).fit(x_train)
monitor.prime(x_reference)
state = monitor.update(x_stream[0])
```

The training data must not also appear in `x_reference`. Reference-set length
and membership should be fixed without inspecting monitoring evidence.

### Published fixed-split demonstration remains supported

The fixed-split code shown in the accompanying nonconform paper remains valid
API usage and continues to run unchanged:

```python
for x_t in data_stream:
    p_t = detector.compute_p_value(x_t)
    state = martingale.update(p_t)
```

This is an illustrative evidence path, not the sequential randomized-rank
construction. `compute_p_value()` intentionally keeps its documented
fixed-calibration, pointwise semantics. Do not attach a Ville anytime guarantee
to this path unless the resulting p-value sequence has a separate conditional
validity argument.

## Minimal Example Notebook

A runnable notebook example is available at:

- `examples/exchangeability_martingale.ipynb`

Open it with Jupyter:

```bash
jupyter notebook examples/exchangeability_martingale.ipynb
```

It uses:

- `oddball` credit-card fraud data
- `IsolationForest` for base anomaly scoring
- `ExchangeabilityMonitor` to produce sequential randomized-rank p-values
- `PowerMartingale`, `SimpleMixtureMartingale`, and `SimpleJumperMartingale`
  for online evidence updates

The example trains on a subset and processes the remaining data in a streaming
loop while logging p-values and evidence statistics step by step.

## Available Martingales

### PowerMartingale

Uses $r_n = \epsilon \cdot p_n^{\epsilon - 1}$ for $\epsilon \in (0, 1]$.

```python
from nonconform.martingales import PowerMartingale

martingale = PowerMartingale(epsilon=0.5)
```

### SimpleMixtureMartingale

Averaged (discrete) mixture over a grid of power martingales.

```python
from nonconform.martingales import SimpleMixtureMartingale

martingale = SimpleMixtureMartingale(epsilons=[0.25, 0.5, 0.75, 1.0])
```

### SimpleJumperMartingale

Implements the Simple Jumper update scheme from conformal martingale literature.

```python
from nonconform.martingales import SimpleJumperMartingale

martingale = SimpleJumperMartingale(jump=0.01)
```

## Alarm Semantics

Alarms are disabled by default.

Set thresholds with `AlarmConfig`:

- `ville_threshold`: threshold on martingale `M_n`
- `restarted_ville_threshold`: threshold on the restarted mixture e-process
- `cusum_threshold`: threshold on the CUSUM/e-CUSUM evidence statistic
- `shiryaev_roberts_threshold`: threshold on the Shiryaev-Roberts evidence
  statistic

`MartingaleState.triggered_alarms` is a tuple of alarm names (for example,
`("ville", "restarted_ville")`) indicating which thresholds are currently
exceeded.
It can be empty when no alarms are active.

`MartingaleState.e_value` and `MartingaleState.log_e_value` expose the current
betting factor $e_n=M_n/M_{n-1}$ in linear and log scale. `MonitorState` also
exposes these values directly.

### Interpreting `ville_threshold`

For a valid nonnegative martingale started at 1 under the null (exchangeability),
Ville's inequality gives:

$$
\Pr\left(\sup_t M_t \ge \lambda\right) \le \frac{1}{\lambda}.
$$

So choosing `ville_threshold = lambda` controls the probability of ever crossing
that threshold on a null stream at most `1 / lambda`.

Example mappings:

- `ville_threshold = 20` -> false alarm probability at most `0.05`
- `ville_threshold = 100` -> false alarm probability at most `0.01`

### Interpreting `restarted_ville_threshold`

`restarted_ville_threshold` applies to a restarted mixture e-process. It uses a
proper weighted sum over possible restart times rather than the raw CUSUM
maximum. This can improve sensitivity to later changes while preserving the same
Ville-style anytime false-alarm probability control as the product martingale.
It is not uniformly more powerful: later restart times receive progressively
smaller prior mass.

Use the same threshold mapping:

```python
alpha = 0.01
alarm_config = AlarmConfig(
    restarted_ville_threshold=1 / alpha,
    cusum_threshold=None,
)
```

The restarted mixture uses the harmonic restart prior
`pi_t = 1 / (t * (t + 1))` with tail mass `1 / (t + 1)`. The tail mass is part
of the e-process accounting and keeps the process initialized at 1.

### Interpreting CUSUM and Shiryaev-Roberts Thresholds

`cusum_threshold` applies to the CUSUM/e-CUSUM statistic. It is useful as a
changepoint evidence statistic, but it is not a Ville threshold. Interpret it
through ARL/FAR guarantees or empirical calibration unless a separate theorem is
provided for the exact implementation.

`shiryaev_roberts_threshold` applies to the SR/e-SR statistic. Depending on the
procedure, SR variants can have ARL/e-detector interpretations, but this
threshold should not be documented as a probability-of-ever-crossing Ville
control unless the implemented statistic is itself an e-process.

Scope of this guarantee:

- `ville_threshold` and `restarted_ville_threshold` provide anytime false-alarm
  control per stream (single null) when the input e-values are conditionally
  valid.
- FDR control across many simultaneous hypotheses or streams requires separate
  multiple-testing procedures; see [FDR Control](fdr_control.md).
- If ordinary and restarted Ville alarms are both enabled and action is taken
  when either fires, allocate error across them or apply a union bound. Giving
  each alarm threshold `1 / alpha` does not make their union an `alpha` test.
- Resetting or retraining starts a new monitoring episode. Repeated episodes
  require alpha spending or another repeated-testing guarantee for lifetime
  false-alarm control.

## Practical Notes

- Keep detector retraining logic outside the martingale classes.
- Interpret alarms as evidence signals, not automated retraining decisions.
- Exact exchangeability-martingale validity follows the sequential conformal
  setup implemented by `SequentialRankConformalizer`. If you reuse a fixed
  calibration ECDF to score a stream, treat alarms as monitoring signals unless
  you have separately justified the resulting p-value sequence.
- If temporal dependence is strong, p-value validity can degrade; monitor model and
  data assumptions alongside evidence statistics.

## References

- **Volkhonskiy, D., Burnaev, E., Nouretdinov, I., Gammerman, A., &
  Vovk, V. (2017)**.
  *[Inductive Conformal Martingales for Change-Point Detection](https://proceedings.mlr.press/v60/volkhonskiy17a.html)*.
  Proceedings of Machine Learning Research, 60, 132-153.
- **Vovk, V., Petej, I., Nouretdinov, I., Ahlberg, E., Carlsson, L., &
  Gammerman, A. (2021)**.
  *[Retrain or not retrain: conformal test martingales for change-point detection](https://proceedings.mlr.press/v152/vovk21b.html)*.
  Proceedings of Machine Learning Research, 152, 191-210.
- **Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023)**.
  *[Game-theoretic statistics and safe anytime-valid inference](https://arxiv.org/abs/2210.01948)*.
  Statistical Science, 38(4), 576-601.
- **Shafer, G., & Vovk, V. (2008)**.
  *[A Tutorial on Conformal Prediction](https://jmlr.org/papers/v9/shafer08a.html)*.
  Journal of Machine Learning Research, 9, 371-421.
