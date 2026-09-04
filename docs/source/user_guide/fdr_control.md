---
description: "Use batch FDR control, derandomized conformal e-values, weighted conformalized selection, simultaneous post-hoc FDP bounds, and online FDR with their distinct assumptions."
---

# False discovery rate control

When a batch contains many candidate anomalies, thresholding every p-value at
the same nominal level ignores multiplicity. False discovery rate (FDR) control
targets the quality of the selected set instead.

Let `R` be the number of discoveries and `V` the number of selected true nulls.
The realized false discovery proportion (FDP) is

$$
\operatorname{FDP}=\frac{V}{\max(R,1)},
$$

and the false discovery rate is

$$
\operatorname{FDR}=\mathbb{E}[\operatorname{FDP}].
$$

The expectation is over repetitions of the complete data-generating and
selection procedure. FDP is observable on one fully labeled family; FDR is not.

!!! note "What `alpha=0.05` means"

    Under the assumptions of the p-values and selection procedure, the
    expected fraction of false discoveries among all discoveries is bounded by
    the stated target, here 0.05. It does not mean that each
    selected point has a 95% probability of being anomalous, nor that at most
    5% of all normal observations will be selected in every realized batch.

## Standard batch workflow

For an unweighted `ConformalDetector`, `select(...)` computes p-values and
applies Benjamini-Hochberg (BH) adjustment to the complete input batch.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(3_000, 4))
x_family = np.vstack(
    [rng.normal(size=(195, 4)), rng.normal(loc=4.5, size=(5, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)

selected = detector.select(x_family, alpha=0.05)
result = detector.last_result
assert result is not None
assert result.p_values is not None

print(f"discoveries: {selected.sum()}")
print("smallest p-values:", np.sort(result.p_values)[:5])
```

The input to one `select(...)` call is one testing family. Calling it separately
on several chunks applies separate procedures. It does not reproduce BH on the
combined p-values.

## How BH selects

For sorted p-values
`p_(1) <= ... <= p_(m)`, BH finds the largest `k` satisfying

$$
p_{(k)}\le \frac{k}{m}\alpha
$$

and rejects the hypotheses with p-values no greater than `p_(k)`. Equivalently,
software can return BH-adjusted p-values and select those no greater than
`alpha`.

Under independent null p-values, or suitable positive regression dependence,
BH controls FDR. Conformal p-values computed against a shared calibration set
are generally dependent. Bates et al. establish the needed positive dependence
and BH control for their conformal outlier-testing construction; applying that
result requires matching its exchangeability and score-construction conditions.

!!! warning "A procedure cannot repair invalid p-values"

    BH, BY, WCS, and online FDR each have assumptions. No multiplicity method
    repairs leakage, a changing score map, a misspecified null population, or a
    calibration-to-test shift outside its model.

## Manual BH or BY adjustment

Use SciPy when you intentionally want adjusted p-values or Benjamini-Yekutieli
(BY). SciPy documents BH for independent or positively regression-dependent
p-values and BY as a more conservative option valid under arbitrary dependence.

```python
import numpy as np
from scipy.stats import false_discovery_control

p_values = np.array([0.001, 0.009, 0.03, 0.08, 0.20, 0.70])
alpha = 0.05

bh_adjusted = false_discovery_control(p_values, method="bh")
by_adjusted = false_discovery_control(p_values, method="by")

print("BH selections:", bh_adjusted <= alpha)
print("BY selections:", by_adjusted <= alpha)
```

BY protects against a broader dependence class by sacrificing power. It does
not protect against choosing the family, method, or target after inspecting the
same results.

Use family-wise error rate control instead when the objective is the
probability of making even one false rejection, rather than the expected
proportion of errors among discoveries.

## Why naive pointwise thresholding fails

If 1,000 true-null p-values are valid and super-uniform, the expected number at
or below `0.05` is at most 50. Equality requires exact uniformity. Selecting all
such p-values would therefore create false discoveries even though every
individual test uses a familiar threshold.

Multiplicity control coordinates the thresholds across the family. It does
not change the meaning of the underlying p-values.

## Weighted conformalized selection

Under covariate shift, weighted conformal p-values need not have the positive
dependence used by ordinary BH. In weighted mode, `select(...)` dispatches to
weighted conformalized selection (WCS), not BH.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.fdr import Pruning

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(2_000, 3))
x_shifted_family = np.vstack(
    [
        rng.normal(loc=0.5, size=(18, 3)),
        rng.normal(loc=6.0, size=(2, 3)),
    ]
)

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.4),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
).fit(x_reference)

selected = detector.select(
    x_shifted_family,
    alpha=0.1,
    pruning=Pruning.DETERMINISTIC,
)
print("selected indices:", np.flatnonzero(selected))
```

The pruning options are:

| Mode | Randomness | Interpretation |
|---|---|---|
| `Pruning.DETERMINISTIC` | None | Deterministic pruning based on WCS rejection-set sizes |
| `Pruning.HOMOGENEOUS` | One shared uniform draw | Randomized pruning with common randomness |
| `Pruning.HETEROGENEOUS` | Independent uniform draws | Randomized pruning with observation-specific randomness |

Pass `seed` to `select(...)` for reproducible randomized pruning. The
detector's seed is used when the selection seed is omitted.

WCS validity additionally requires the weighted-conformal covariate-shift
assumptions and suitable weights. See
[Weighted conformal inference](weighted_conformal.md).

## Derandomized conformal e-values

Split-conformal decisions can vary with the random training/calibration split.
Repeating the split and voting on discoveries does not automatically preserve
FDR control. `conformal_e_value_selection(...)` instead constructs conformal
e-values for each split, averages them uniformly, and applies e-BH once to the
aggregated evidence.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import conformal_e_value_selection

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(3_000, 4))
x_family = np.vstack(
    [rng.normal(size=(195, 4)), rng.normal(loc=4.5, size=(5, 4))]
)

test_scores = []
calib_scores = []
for seed in range(5):
    detector = ConformalDetector(
        detector=IsolationForest(random_state=seed),
        strategy=Split(n_calib=0.3),
        score_polarity="higher_is_normal",
        seed=seed,
    ).fit(x_reference)
    detector.score_samples(x_family)

    result = detector.last_result
    assert result is not None
    assert result.test_scores is not None
    assert result.calib_scores is not None
    test_scores.append(result.test_scores)
    calib_scores.append(result.calib_scores)

e_result = conformal_e_value_selection(
    np.vstack(test_scores),
    np.vstack(calib_scores),
    alpha=0.1,
)

print("selected indices:", np.flatnonzero(e_result.selected))
print("inner alpha:", e_result.alpha_bh)
```

This is an expert workflow for one fixed test family analyzed by repeated valid
split-conformal score maps. Use it when single-split decisions are unstable,
not as a default replacement for `detector.select(...)`. The default inner
threshold is `alpha_bh=alpha / 10`; pass it explicitly only when the analysis
plan justifies another value.

!!! warning "Guarantee scope"

    Derandomized e-values require exchangeability of the normal reference data
    and null test points, a fixed score orientation in which larger scores mean
    more anomalous, repeated valid split-conformal constructions, and one final
    e-BH application. The package currently provides uniform aggregation, not
    the adaptive weighting described in the paper.

See the complete
[derandomized conformal e-value example](../examples/derandomized_e_values.md).

## Post-hoc simultaneous FDP bounds

FDR control selects by a procedure fixed in advance and controls expected FDP.
`conformal_fdp_upper_bound_from_result(...)` answers a different question: it
constructs a high-confidence upper envelope for realized FDP that is
simultaneous over p-value thresholds. This permits threshold exploration within
the returned certificate's scope.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.fdr import conformal_fdp_upper_bound_from_result

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(5_000, 3))
x_family = np.vstack(
    [rng.normal(size=(180, 3)), rng.normal(loc=5.0, size=(20, 3))]
)

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_reference)
detector.compute_p_values(x_family)

certificate = conformal_fdp_upper_bound_from_result(
    detector.last_result,
    confidence=0.95,
    n_resamples=500,
    seed=42,
    thresholds=np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.1]),
)

print(certificate.to_frame())
print("FDP upper bound at 0.05:", certificate.bound_at(0.05))
```

The result-based API deliberately accepts only unweighted `Split` or detached
calibration results with `Empirical` p-values. It rejects weighted,
probabilistic/KDE, conditional-calibration, and resampling-strategy results.
The certificate does not cover changing the detector or model after looking at
the curve.

An upper bound of `1.0` is valid but uninformative, not a build or API failure.
Certificate tightness depends on the calibration count, family size, observed
p-values, confidence level, and envelope method. Do not change those inputs
post hoc solely to obtain a more attractive bound.

Choose the envelope `method` before inspecting the bound. Supported values are
`"mc_thc"`, `"mc_hc"`, `"mc_ks"`, `"ks"`, and `"mc_bj"`.

!!! note "FDR target and FDP confidence are different"

    `alpha` is the target expected false discovery proportion for a selection
    procedure. `confidence=0.95` is the simultaneous coverage probability of
    an FDP upper-bound certificate. Neither can be substituted for the other.

## Repeated and online testing

### Repeated batches

Applying BH at level `alpha` to every batch can control FDR within each batch
under the corresponding assumptions. It does not automatically control the
FDR of all discoveries pooled across time, and it does not create a lifetime
false-alarm bound.

If all observations form one fixed family and can be held until the family is
complete, apply one batch procedure to that family. If hypotheses genuinely
arrive over time and decisions cannot wait, use an online multiple-testing
method whose assumptions match the p-value process.

### Online FDR

The optional `fdr` extra provides the separate `online_fdr` package:

```bash
pip install "nonconform[fdr]"
```

The following is a runnable API demonstration with a simulated p-value stream.
It does not establish that p-values from an arbitrary application satisfy the
method's assumptions.

```python
import numpy as np
from online_fdr import LordThree

rng = np.random.default_rng(42)
p_value_stream = rng.uniform(size=100)
p_value_stream[[30, 70]] = [1e-6, 1e-7]

controller = LordThree(alpha=0.05, wealth=0.025, reward=0.05)
rejections = np.array(
    [controller.test_one(float(p_value)) for p_value in p_value_stream],
    dtype=bool,
)

print(np.flatnonzero(rejections))
```

Online FDR procedures typically require null p-values that are conditionally
super-uniform relative to the information available before each test, plus the
procedure's stated dependence conditions. Repeated
`ConformalDetector.compute_p_value(...)` calls against one fixed calibration
ECDF do not automatically supply that conditional property.

For evidence of a distributional change in one ordered stream, use
[exchangeability martingales](exchangeability_martingales.md). That is a
sequential change-monitoring problem, not automatically an online
multiple-testing problem.

## Choosing the correct lane

| Goal | Use |
|---|---|
| Select anomalies in one fixed batch | `detector.select(batch, alpha=...)` |
| Adjust standard p-values manually | SciPy BH or BY, after checking assumptions |
| Select under modeled covariate shift | Weighted `detector.select(...)`, which uses WCS |
| Stabilize repeated split-conformal selections | `conformal_e_value_selection(...)` and e-BH |
| Explore p-value thresholds with a simultaneous realized-FDP certificate | `conformal_fdp_upper_bound_from_result(...)` |
| Test an open-ended sequence of distinct hypotheses | A justified online FDR procedure |
| Detect loss of exchangeability in one stream | `ExchangeabilityMonitor` and a conformal martingale |

## References

- [Benjamini and Hochberg (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
  introduces BH FDR control.
- [Benjamini and Yekutieli (2001)](https://doi.org/10.1214/aos/1013699998)
  studies FDR control under dependence.
- [Bates et al. (2023)](https://doi.org/10.1214/22-AOS2244)
  analyzes conformal p-values for outlier testing, their dependence, and FDR
  control.
- [Bashari et al. (2023)](https://arxiv.org/abs/2302.07294)
  develops derandomized novelty detection with conformal e-values and e-BH.
- [SciPy `false_discovery_control`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)
  documents the implemented BH and BY adjustments.
- [Jin and Candès (2023)](https://arxiv.org/abs/2307.09291)
  introduces weighted conformal p-values and WCS under covariate shift.
- [Song, Jin, and Candès (2026)](https://arxiv.org/abs/2605.20726)
  develops simultaneous FDP bounds over conformal-p-value thresholds.
- [Javanmard and Montanari (2018)](https://doi.org/10.1214/17-AOS1629)
  develops LORD-style online FDR control.
