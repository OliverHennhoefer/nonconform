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
- `MixtureMartingale`

## Why P-values (Not Raw Scores)

These martingales are conformal/exchangeability tests. Under exchangeability,
properly randomized sequential conformal p-values in the classical construction
are independent `Uniform(0, 1)` variables. This is the property used by the
martingale betting factors.

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

## Basic sequential usage

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_train = rng.standard_normal((300, 5))
x_stream = np.vstack(
    [
        rng.standard_normal((60, 5)),
        rng.normal(loc=3.0, size=(40, 5)),
    ]
)

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
        alarm_config=AlarmConfig(restarted_ville_threshold=20.0)
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
else:
    print("No alarm in this finite stream")
```

`from_split_detector(...)` copies the fitted scoring model and primes the
sequential rank history with its calibration scores. Martingale capital remains
at one until the first monitored observation. The original detector and its
fixed-calibration behavior are unchanged.

Alternatively, fit a monitor directly and supply a separate reference set:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_train = rng.normal(size=(300, 5))
x_reference = rng.normal(size=(100, 5))
x_stream = rng.normal(size=(10, 5))

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
The monitor deep-copies explicitly supplied conformalizers and martingales as
configuration prototypes, so later mutations to the caller's objects cannot
alter the monitored evidence sequence. Reference batches are scored and
validated completely before any scores are committed to rank history.

### Published fixed-split demonstration remains supported

The fixed-split code shown in the accompanying nonconform paper remains valid
API usage and continues to run unchanged:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(400, 3))
data_stream = rng.normal(size=(20, 3))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)
martingale = PowerMartingale(epsilon=0.5)

for x_t in data_stream:
    p_t = detector.compute_p_value(x_t)
    state = martingale.update(p_t)

print(state.martingale)
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
- `PowerMartingale` for online evidence updates

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

### MixtureMartingale

Combines existing martingales using fixed normalized weights. Each expert
receives the same p-value and maintains its own ordinary capital, harmonic
restart evidence, CUSUM, and Shiryaev-Roberts (SR) statistic. The mixture averages
each corresponding statistic separately:

$$
M_t = \sum_j w_j M_{t,j},\quad
E_t^{\mathrm{restart}} = \sum_j w_j E_{t,j}^{\mathrm{restart}},\quad
C_t = \sum_j w_j C_{t,j},\quad R_t = \sum_j w_j R_{t,j}.
$$

For a fitted unweighted `Split` detector:

```python
from nonconform.martingales import AlarmConfig, MixtureMartingale, PowerMartingale
from nonconform.monitoring import ExchangeabilityMonitor

mixture = MixtureMartingale(
    [PowerMartingale(epsilon=0.25), PowerMartingale(epsilon=0.5)],
    weights=[1, 1],
    alarm_config=AlarmConfig(shiryaev_roberts_threshold=1_000),
)
monitor = ExchangeabilityMonitor.from_split_detector(
    detector, martingale=mixture, seed=42
)
```

With valid component e-detectors, this SR threshold gives an average run length
of at least 1,000 observations under the null. It is not a 0.1% lifetime
false-alarm probability. Alarms use only the mixture's `AlarmConfig`; component
alarm names are not forwarded.

Components must be reset at construction and are deep-copied. Equal weights are
the default; supplied finite nonnegative weights are normalized once and stay
fixed. Zero-weight components do not participate. Jumper, custom
`BaseMartingale` implementations, and nested mixtures are supported. The
mixture owns all component state. Aggregation adds O(J) work and storage on top
of the J components' costs.

#### Difference from SimpleMixtureMartingale

`SimpleMixtureMartingale` averages power capitals accumulated since the start,
then feeds that mixture's capital ratios into its alarm recurrences.
`MixtureMartingale` with separate power experts instead averages independently
maintained alarm statistics. The ordinary capitals agree for equal weights and
the same epsilon grid, but their restart, CUSUM, and SR statistics generally do
not. In particular, a mixture of CUSUM statistics is not a CUSUM computed from
mixture-capital ratios.

For example, with experts at `epsilon=0.5` and `epsilon=1`, a long stable history
can leave the all-history mixture dominated by the neutral expert. Subsequent
small p-values then produce almost unit mixture factors. In the per-expert
construction, the `epsilon=0.5` expert's SR recurrence still starts a fresh
candidate every step, and its SR statistic retains its fixed mixture weight.
This addresses that composition mechanism; it does not guarantee uniform
late-change power or remove inertia inside an adaptive component.

Fixed normalized mixtures inherit the corresponding martingale/e-process or
e-detector guarantee only if all participating components satisfy it under a
common null and information history (filtration). The same condition applies
to custom and nested components; numeric validation cannot establish validity.
This is an established mixture construction, described in Section 3 of
[Shin, Ramdas, and Rinaldo](https://arxiv.org/html/2203.03532v4#S3).

If a component raises an error or returns invalid log statistics during an
update, the mixture retains its last completed state and blocks further
updates until `reset()`. Its components may have partially advanced. When used
in a monitor, reset the **whole monitor**, since rank history may also have
advanced. A reset starts a new episode with the error-accounting implications
described below.

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

`MartingaleState.e_value` and `MartingaleState.log_e_value` expose the ordinary
capital ratio $e_n=M_n/M_{n-1}$ in linear and log scale. `MonitorState` also
exposes these values directly. For `MixtureMartingale`, this ratio does not
reproduce its other statistics: SR, CUSUM, and harmonic restart evidence are
averaged independently. Feed the mixture itself to the monitor rather than
reconstructing those statistics from its `e_value` sequence.

For consecutive exact zero capitals or consecutive infinite log-capitals,
`MixtureMartingale` reports the undefined ratio as a neutral diagnostic factor
one (`log_e_value=0`). The combined capital itself remains zero or infinite.
Finite log-capitals retain their actual log ratio even if their linear-scale
values underflow or overflow.

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
from nonconform.martingales import AlarmConfig

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

`cusum_threshold` and `shiryaev_roberts_threshold` apply to CUSUM/e-CUSUM and
SR/e-SR statistics. For the implemented recurrences with conditionally valid
nonnegative betting increments, each is an e-detector. At threshold $A>1$, the
first crossing time $T$ satisfies the average-run-length (ARL) bound

$$
\mathbb{E}_{\infty}[T] \ge A.
$$

The same bound applies to fixed normalized mixtures of component e-detectors.
This follows from the e-detector stopping-time property; see
[Shin, Ramdas, and Rinaldo, Sections 2–3](https://arxiv.org/html/2203.03532v4#S2).
It does **not** imply that the probability of ever raising a false alarm is at
most $1/A$. An ARL lower bound also does not promise the same remaining waiting
time conditional on having survived to an arbitrary age.

For the classical randomized-rank conformal construction, the required
filtration is the past conformal p-values (and an independently fixed training
construction), not arbitrary access to the raw score/reference history. Betting
choices must be predictable from the permitted past information. Custom
components using additional information require their own argument. Ordinary
fixed-calibration marginal p-values do not suffice. If these conditions are not
established, the raw thresholds alone confer no theoretical ARL guarantee.

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
- The sequential conformal construction implemented by
  `SequentialRankConformalizer` supports the exact exchangeability-martingale
  argument when its stated exchangeability, frozen-scorer, and randomization
  assumptions hold. If you reuse a fixed calibration ECDF to score a stream,
  treat alarms as monitoring signals unless you have separately justified the
  resulting p-value sequence.
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
- **Shin, J., Ramdas, A., & Rinaldo, A. (2024)**.
  *[E-detectors: a nonparametric framework for sequential change detection](https://arxiv.org/abs/2203.03532)*.
  New England Journal of Statistics in Data Science, 2(2), 229-260.
