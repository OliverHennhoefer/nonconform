# Exchangeability Martingales

Monitor streaming conformal p-values for evidence against exchangeability.

## What This Feature Does

`nonconform.martingales` consumes sequential p-values and maintains:

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

- under exchangeability, conformal p-values are approximately i.i.d.
  `Uniform(0, 1)` (independent and uniformly distributed)

Raw anomaly scores do not satisfy this requirement directly. Use `ConformalDetector`
to produce p-values first, then feed those p-values into a martingale.

## Basic Usage

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale

rng = np.random.default_rng(42)
x_train = rng.standard_normal((300, 5))
x_stream = rng.standard_normal((100, 5))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.2),
    score_polarity="auto",
    seed=42,
)
detector.fit(x_train)

martingale = SimpleJumperMartingale(
    alarm_config=AlarmConfig(
        ville_threshold=100.0,
        restarted_ville_threshold=100.0,
    )
)

for x_t in x_stream:
    p_t = detector.compute_p_values(x_t.reshape(1, -1))[0]
    state = martingale.update(p_t)
    if "restarted_ville" in state.triggered_alarms:
        print(
            "Restarted Ville alarm "
            f"at step={state.step}, M={state.restarted_martingale:.2f}"
        )
        break
```

## Minimal Example Notebook

A runnable notebook example is available at:

- `examples/martingale_streaming.ipynb`

Open it with Jupyter:

```bash
jupyter notebook examples/martingale_streaming.ipynb
```

It uses:

- `sklearn.datasets.load_iris`
- `IsolationForest` for base anomaly scoring
- `ConformalDetector` to produce streaming p-values
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
maximum. This gives CUSUM-like sensitivity to late changes while preserving the
same Ville-style anytime false-alarm probability control as the product
martingale.

Use the same threshold mapping:

```python
alpha = 0.01
alarm_config = AlarmConfig(
    ville_threshold=1 / alpha,
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

## Practical Notes

- Keep detector retraining logic outside the martingale classes in this release.
- Interpret alarms as evidence signals, not automated retraining decisions.
- If temporal dependence is strong, p-value validity can degrade; monitor model and
  data assumptions alongside evidence statistics.
