---
description: "Understand conformal anomaly scores, empirical p-values, exchangeability, conditional calibration, multiple testing, and sequential ranks."
---

# Understanding conformal inference

Conformal inference turns a detector's relative anomaly scores into p-values by
ranking test scores against scores from suitable reference observations. The
result separates two concerns:

- the detector determines which observations look unusual; and
- the conformal layer calibrates how extreme those scores are under a stated
  null and data-exchangeability design.

!!! abstract "Two primary workflows"

    For a fixed batch, compute conformal p-values and use a justified
    multiple-testing procedure, usually `select(...)`. For an ordered stream,
    construct randomized sequential ranks and accumulate them with an
    exchangeability martingale. Batch p-values and sequential p-values are not
    interchangeable.

## Scores, p-values, and decisions

| Quantity | What it represents | What it does not represent |
|---|---|---|
| Anomaly score | Detector-specific ordering of unusualness | A calibrated probability |
| Conformal p-value | Rank-based evidence against the reference null | Posterior probability that the point is normal or anomalous |
| FDR-controlled selection | A decision within a declared testing family | A guarantee for each selected point |
| Martingale value | Cumulative sequential evidence against exchangeability | An FDR estimate for a batch |

A raw threshold such as `score > 2.5` has detector- and dataset-specific
meaning. A conformal p-value uses the calibration distribution to put that
score on a rank scale. Its validity still depends on the data roles and
assumptions described below.

## A complete split-conformal batch

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(1_000, 4))
x_family = np.vstack(
    [rng.normal(size=(48, 4)), rng.normal(loc=4.5, size=(2, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

selected = detector.select(x_family, alpha=0.1)
result = detector.last_result
assert result is not None
assert result.p_values is not None

print("p-values:", result.p_values[:5])
print("discoveries:", np.flatnonzero(selected))
```

`Split` randomly reserves 30% of `x_reference` for calibration and fits the
base detector on the remaining rows. The resulting score map is then fixed for
both calibration and test scoring. In unweighted mode, `select(...)` applies
BH to the p-values from the complete input family.

## Classical empirical p-values

Assume larger normalized scores mean more anomalous. Let
`S_1, ..., S_n` be calibration scores and `S(x)` a true-null test score. The
default `Empirical(tie_break="classical")` computes

$$
p(x)=\frac{1+\sum_{i=1}^{n}\mathbf{1}\{S_i\ge S(x)\}}{n+1}.
$$

The leading `1` accounts for the test observation itself. Counting ties as at
least as extreme makes the p-value deterministic and conservative.

Under exchangeability of the calibration scores and a true-null test score,
conditional on a score map fitted independently of them,

$$
\Pr\{p(x)\le a\}\le a
$$

for every `a` in `[0, 1]`. This is super-uniformity. It is a repeated-sampling
statement, not a posterior interpretation of the realized p-value.

### Resolution

Classical empirical p-values lie on a grid with step `1 / (n + 1)`. The
smallest possible value is therefore `1 / (n + 1)`.

```python
import numpy as np

from nonconform import Empirical

calibration_scores = np.array([0.1, 0.3, 0.3, 0.8, 1.1])
test_scores = np.array([0.0, 0.3, 2.0])

p_values = Empirical().compute_p_values(test_scores, calibration_scores)
print(p_values)
print("grid step:", 1 / (len(calibration_scores) + 1))
```

A coarse grid can limit power in a large multiple-testing family. It does not
make the p-values invalid.

## Randomized tie handling

`Empirical(tie_break="randomized")` uses an independent
`U ~ Uniform(0, 1)` for each test score:

$$
p(x)=\frac{
\#\{i:S_i>S(x)\}+U\left(\#\{i:S_i=S(x)\}+1\right)
}{n+1}.
$$

This interpolates tied calibration mass and the test point's own rank. It can
produce values below the classical resolution floor. Set `seed` on
`ConformalDetector` for reproducibility.

Randomization is part of the statistical procedure. It should not be replaced
with a fixed constant merely to obtain smoother-looking values.

## Exchangeability

A finite sequence `Z_1, ..., Z_m` is exchangeable if, for every permutation
`pi`,

$$
(Z_1,\ldots,Z_m)\stackrel{d}{=}
(Z_{\pi(1)},\ldots,Z_{\pi(m)}).
$$

Independent and identically distributed observations are exchangeable, but
exchangeability does not require independence. Operationally, it rules out
systematic order, source, sampling, or processing differences that make some
positions statistically distinguishable.

For split conformal outlier testing, the important comparison is between
calibration observations and each true-null test observation after conditioning
on the fixed training-only scoring construction. Test anomalies need not follow
the null distribution; they are the alternatives the procedure seeks.

### What can violate it

- temporal drift or seasonality not represented in calibration;
- different sensors, sites, users, or sampling policies;
- fitting preprocessing on calibration or test data;
- tuning the score map against the final testing family;
- duplicated entities or dependence that does not satisfy the selected
  multiple-testing theorem;
- filtering rows based on information unavailable under the declared protocol.

Shuffling a nonexchangeable dataset does not make it exchangeable. Marginal
two-sample tests can reveal discrepancies but cannot prove joint
exchangeability.

## Integrated and detached calibration

`fit(...)` lets a strategy create fitting and calibration roles. With `Split`,
you can instead own those roles explicitly: fit the base detector on proper
training data, then call `calibrate(...)` with a separate calibration array.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_fit = rng.normal(size=(400, 3))
x_calibration = rng.normal(size=(200, 3))
x_test = np.vstack(
    [rng.normal(size=(18, 3)), rng.normal(loc=4.0, size=(2, 3))]
)

base_detector = IsolationForest(random_state=42).fit(x_fit)
detector = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.3),
    seed=42,
)
detector.calibrate(x_calibration)

print(detector.compute_p_values(x_test))
```

Detached calibration currently supports `Split` only. The `n_calib` value is
not used to split the explicit calibration array.

## Several test points and dependence

Each standard split-conformal p-value can be marginally valid while p-values
for several test points remain dependent through their shared calibration set.
Multiple-testing validity therefore requires both valid null p-values and the
dependence assumptions of the selected procedure.

Bates et al. show positive dependence and exact BH FDR control for their
conformal outlier-testing construction. Match the theorem's score construction
and exchangeability conditions before citing it. For a broader dependence
class, BY is more conservative; it still cannot correct invalid p-values.

See [False discovery rate control](fdr_control.md) for the procedures and their
distinct targets.

## Conditionally calibrated p-values

Standard conformal p-values are marginally valid over the random calibration
set and test point. Reusing one realized calibration set for many future test
points motivates a stronger target: with probability at least `1 - delta` over
the calibration draw, future true-null p-values are super-uniform conditional
on that calibration set. Conditional independence of future test points can
then support stronger multiple-testing arguments.

`ConditionalEmpirical` first computes empirical conformal p-values and applies
a calibration map

$$
\widetilde p=C_{n,\delta}(p).
$$

It follows the maps in the reference implementation accompanying Bates et al.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.scoring import ConditionalEmpirical

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(700, 4))
x_family = np.vstack(
    [rng.normal(size=(15, 4)), rng.normal(loc=4.5, size=(5, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    estimation=ConditionalEmpirical(
        method="simes",
        delta=0.1,
        tie_break="classical",
    ),
    seed=42,
).fit(x_reference)

selected = detector.select(x_family, alpha=0.1)
result = detector.last_result
assert result is not None
assert result.p_values is not None

print("minimum conditional p-value:", result.p_values.min())
print("discoveries:", selected.sum())
```

`delta` belongs to the conditional-calibration event. `alpha` belongs to the
downstream selection rule. They need not be equal.

### Available maps

| Method | Construction | Implementation detail |
|---|---|---|
| `"dkwm"` | Finite-sample DKW-Massart concentration boundary | Deterministic and often conservative |
| `"simes"` | Finite-sample Simes sequence | `simes_kden` sets `k = floor(n_cal / simes_kden)`, bounded below by one |
| `"mc"` | Hybrid boundary with a Monte Carlo-estimated finite-sample correction | Correction is cached for a fixed `(n_cal, delta)`; `mc_num_simulations` controls numerical effort |
| `"asymptotic"` | Iterated-log asymptotic boundary | An approximation rather than a finite-sample Monte Carlo calibration |

For fewer than 17 calibration scores, `"mc"` and `"asymptotic"` fall back to
`"dkwm"` because their iterated-log constants are not defined. Method choice is
a statistical and power tradeoff; do not tune it on the final testing family.

`ConditionalEmpirical` supports unweighted p-values only. It is not exported
from the package root; import it from `nonconform.scoring`.

## Probabilistic p-value estimation

`Probabilistic` fits a kernel density estimate to calibration scores and
returns a continuous estimated survival probability. It can help when a smooth
model of the score distribution is appropriate, but it does not inherit the
exact rank-based finite-sample guarantee of `Empirical`.

Install it with `nonconform[probabilistic]`, state the KDE modeling assumption,
and validate calibration on untouched null data. Do not present continuous
resolution itself as evidence of better validity.

## Resampling strategies

`CrossValidation` and `JackknifeBootstrap` construct calibration scores without
reserving one fixed holdout, and plus mode aggregates test scores across
retained models. These can be useful data-efficiency and stability tools, but
their package-specific anomaly-score aggregation does not automatically inherit
all prediction-interval theorems associated with CV+, jackknife+, or JaB+.

See [Conformalization strategies](conformalization_strategies.md) for exact
mechanics and theorem scope.

## Sequential conformal ranks

A fixed calibration ECDF gives marginal batch p-values. Repeatedly calling it
on a stream does not automatically produce the conditionally valid p-value
sequence required by a conformal martingale.

`ExchangeabilityMonitor` instead ranks each new score among the sequential
score history with independent randomized tie breaking, then feeds that
p-value to a betting martingale.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform.martingales import AlarmConfig, PowerMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_fit = rng.normal(size=(300, 3))
x_prime = rng.normal(size=(150, 3))
x_stream = np.vstack(
    [rng.normal(size=(20, 3)), rng.normal(loc=3.0, size=(15, 3))]
)

monitor = ExchangeabilityMonitor(
    detector=IsolationForest(random_state=42),
    martingale=PowerMartingale(
        epsilon=0.5,
        alarm_config=AlarmConfig(ville_threshold=20.0),
    ),
    seed=42,
)
monitor.fit(x_fit).prime(x_prime)
states = monitor.update_many(x_stream)

print("final martingale:", states[-1].martingale)
print("final alarms:", states[-1].triggered_alarms)
```

The null requirement is exchangeability of priming and monitored scores
conditional on the frozen training-only scorer. Do not refit during an episode.
See [Exchangeability martingales](exchangeability_martingales.md) for Ville,
restart mixtures, CUSUM, and Shiryaev-Roberts interpretations.

## Interpreting outputs correctly

- Smaller p-values mean greater incompatibility with the calibration null in
  the direction encoded by the normalized score.
- `score_samples(...)` returns aggregated raw scores, not p-values.
- `compute_p_value(...)` handles one unweighted observation; weighted mode
  requires a representative batch.
- `compute_p_values(...)` returns a NumPy array for NumPy input and an
  index-preserving pandas `Series` for pandas input.
- `select(...)` returns the corresponding Boolean mask and caches the p-values
  from the same pass in `last_result`.
- A p-value of `0.01` is not a 1% probability that the point is normal.
- An FDR target is not a per-observation posterior confidence.

## Validity checklist

- Define the null population before examining the test family.
- Fit the detector and learned preprocessing without calibration or test
  leakage.
- Confirm calibration and true-null test observations satisfy the required
  exchangeability or weighted-shift design.
- Verify score polarity.
- Calculate empirical p-value resolution from the actual calibration count.
- Match the multiple-testing procedure to the p-value dependence structure.
- Keep batch, conditional-calibration, weighted, and sequential guarantees
  distinct.
- Evaluate calibration and power on data not used to choose the procedure.

## References

- [Vovk, Gammerman, and Shafer (2005), *Algorithmic Learning in a Random World*](https://doi.org/10.1007/b106715)
  develops the foundational conformal framework.
- [Shafer and Vovk (2008), *A Tutorial on Conformal Prediction*](https://jmlr.org/papers/v9/shafer08a.html)
  provides an accessible formal tutorial.
- [Lei et al. (2018), *Distribution-Free Predictive Inference for Regression*](https://doi.org/10.1080/01621459.2017.1307116)
  presents split conformal inference and related methods.
- [Bates et al. (2023), *Testing for Outliers with Conformal p-values*](https://doi.org/10.1214/22-AOS2244)
  analyzes marginal and conditionally calibrated conformal p-values and their
  multiple-testing behavior.
- [Conditional conformal p-values reference implementation](https://github.com/msesia/conditional-conformal-pvalues)
  accompanies Bates et al.
- [Volkhonskiy et al. (2017), *Inductive Conformal Martingales for Change-Point Detection*](https://proceedings.mlr.press/v60/volkhonskiy17a.html)
  develops inductive conformal martingales for change detection.
