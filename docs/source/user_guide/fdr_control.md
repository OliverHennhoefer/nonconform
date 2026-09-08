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

For an unweighted `ConformalDetector` using a p-value strategy such as `Split`,
`select(...)` computes p-values and applies Benjamini-Hochberg (BH) adjustment
to the complete input batch. `DerandomizedSplits` instead uses the e-value
procedure described below.

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

`DerandomizedSplits` fits repeated training/calibration splits, constructs
conformal e-values separately for every split, averages the evidence uniformly,
and applies e-BH once. It manages the replicas and calibration rows internally.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, DerandomizedSplits

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(3_000, 4))
x_test = np.vstack(
    [rng.normal(size=(195, 4)), rng.normal(loc=4.5, size=(5, 4))]
)

detector = ConformalDetector(
    detector=IsolationForest(),
    strategy=DerandomizedSplits(n_repetitions=5, n_calib=0.3),
    seed=42,
).fit(x_reference)
mask = detector.select(x_test, alpha=0.1)
e_result = detector.last_selection_result
assert e_result is not None
print("selected indices:", np.flatnonzero(mask))
print("inner alpha:", e_result.alpha_bh)
```

Set advanced options on the strategy: `alpha_bh=None` uses `alpha / 10`, and
`tie_seed=None` automatically derives a tie stream separate from fitting.
An explicit `tie_seed` overrides only tie handling. With a detector seed, the
splits and randomized secondary ranks are reproducible for deterministic
scoring. Without one, randomness is generated once per fit. Fix the procedure
and inner threshold before inspecting test evidence.

`select()` returns a boolean mask and stores a defensive `EValueSelectionResult`
in `last_selection_result`. Its fields include `e_values`, `selected`,
`e_threshold`, `alpha`, `alpha_bh`, `n_repetitions`, `n_calibration`, and the
effective `tie_seed`. Raw scoring and refitting clear selection diagnostics.
`last_result` is `None` after e-value selection.

`score_samples()` still aggregates raw scores using `aggregation`, but selection
always constructs evidence from the separate model/calibration pairs. P-value
methods are unsupported for this strategy. Use unweighted integrated fitting;
non-Empirical estimation is rejected because p-value estimation is unused.

For existing individual split snapshots, `select_conformal_e_values(...)`
remains available. It checks native provenance and identical test-batch content
and ordering, but cannot track modifications to snapshot score arrays. Its
default `tie_seed=None` still rejects tied scores.

!!! warning "Guarantee scope"

    Requires exchangeable normal reference and null test observations, the same
    fixed test family in the same row order for every repetition, valid
    integrated unweighted splits, and one final e-BH application. Randomized
    secondary ranks provide strict ordering without perturbing unequal scores.
    The aggregate null-evidence condition supports e-BH; individual constructed
    values need not be ordinary e-values. Uniform aggregation is the only
    supported evidence aggregation. Realized FDP in one batch is not expected
    FDR, and aggregation is not a promise of improved stability on every dataset.

See the [Shuttle example and advanced result-list workflow](../examples/derandomized_e_values.md).

## Post-hoc simultaneous FDP bounds

FDR control selects by a procedure fixed in advance and controls expected FDP.
`detector.fdp_bounds(x, ...)` answers a different question: it
constructs a high-confidence upper envelope for realized FDP that is
simultaneous over p-value thresholds. This permits threshold exploration within
the returned certificate's scope.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

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
certificate = detector.fdp_bounds(
    x_family,
    confidence=0.95,
    n_resamples=500,
    seed=42,
)

print(certificate.to_frame(thresholds=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]))
print("FDP upper bound at 0.05:", certificate.bound_at(0.05))

threshold = certificate.threshold_for(max_fdp=0.10)
selected = (
    certificate.select(threshold)
    if threshold is not None
    else np.zeros(certificate.n_test, dtype=bool)
)
print("discoveries at target FDP 0.10:", selected.sum())
```

`threshold_for(max_fdp=...)` finds the largest observed p-value cutoff whose
bound is at most the requested target. This maximizes discoveries among
qualifying threshold selections. It checks the entire observed curve because
FDP bounds need not be monotone. The target must be a finite numeric scalar in
`[0, 1]`; booleans and vectors are rejected. `None` means no nonempty selection
qualifies. Use `is not None`: zero is a valid cutoff. Queries reuse prepared
certificate state without rescoring, sorting again, or resampling.

The detector and snapshot APIs require native provenance for unweighted `Split`
inference with `Empirical` p-values, including detached calibration. They reject
unknown provenance, weighted, probabilistic/KDE, conditional, CV/bootstrap, and
`DerandomizedSplits` scopes. Native checks assume unmodified snapshots: they
check scope and batch dimensions, but cannot prove array integrity or establish
scientific exchangeability. The reference construction requires a fixed scoring
map and exchangeability of the calibration and null test observations.

Native certification also requires a recognized empirical tie mode recorded
when the p-values were computed. This private provenance is immutable and is
preserved in defensive snapshots. Classical empirical p-values support tied
scores and do not require stored test scores. Randomized empirical p-values
require finite test scores of the correct shape and reject exact score ties
between the calibration and test sets. Duplicates confined to either set are
allowed; equal p-values alone are not evidence of cross-set score ties.

An existing snapshot can be certified without scoring again:

```python
result = detector.last_result
certificate = result.fdp_bounds(confidence=0.95, seed=42)
mask = certificate.select(0.05)  # p-value cutoff, not a target FDP
```

`select()` returns a Boolean NumPy mask in the original observation order,
including for pandas input. The certificate owns immutable evidence and prepared
query state. Editing the source snapshot, refitting the detector, or editing a
returned table cannot change it. `to_frame()` defaults to sorted unique observed
p-values; custom grids preserve order and duplicates. Scalar/vector `bound_at()`
and `precision_at()` reuse the realized envelope without resampling.

For external p-values, the explicit expert route is:

```python
from nonconform.fdr import FDPCertificate

certificate = FDPCertificate.from_p_values(
    result.p_values, n_calibration=len(result.calib_scores), seed=42
)
```

The caller must verify the expert route's provenance and statistical assumptions,
including the supported score and tie construction described below. P-values
alone cannot establish tie provenance; the expert factory does not bypass
these statistical requirements.
None of these entry points covers changing the detector, scoring map, or testing
family after inspecting the curve.

### Why selecting a target FDP is valid

Write the simultaneous bound as `B(t)` and let `delta = 1 - confidence`. On the
event that `FDP(t) <= B(t)` at every threshold, any chosen threshold with
`B(t) <= max_fdp` also has `FDP(t) <= max_fdp`. Selecting nothing when no cutoff
qualifies has FDP zero. Consequently, under the certificate's assumptions,

$$
\Pr\{\operatorname{FDP}(\widehat{t}) > \texttt{max\_fdp}\}
\le 1 - \texttt{confidence}.
$$

Searching the curve needs no additional multiplicity correction. At target
`0.10` and confidence `0.95`, the probability of realized FDP exceeding 10% is
at most 5%. This is different from expected-FDR control at 10%.

[Song, Jin, and Candès (2026), Theorem 5.4 and Proposition 5.5](https://arxiv.org/html/2605.20726v2#S5)
provide the conformal simultaneous envelope and its self-refinement. Threshold
inversion above is a direct consequence of simultaneous coverage.
[Goeman and Solari (2011)](https://doi.org/10.1214/11-STS356) establish the general
post hoc inference principle with a different construction. The conformal
reference is an arXiv preprint.

The paper assumes no score ties. The supported extension to tied scores follows
by assigning independent continuous secondary keys to all observations,
ordering first by score and then by key, and constructing auxiliary randomized
conformal ranks with independent uniform offsets. Exchangeability is preserved.
Classical empirical p-values are at least these auxiliary ranks, so their null
rejection counts are bounded by the auxiliary counts. Randomized empirical
p-values coincide with the auxiliary ranks when no calibration/test score ties
occur, regardless of duplicates confined to either set. This is why the native
randomized workflow checks cross-set score equality.

For Monte Carlo methods, coverage averages jointly over the data, empirical
randomization when applicable, and independent envelope simulations. A seed
reproduces a draw; it does not establish coverage conditional on that draw.
The randomized tie check likewise does not establish coverage conditional on
passing the check. If certification is rejected, make no certified discoveries;
repeatedly changing the data or seed until certification succeeds is outside
this guarantee.

An upper bound of `1.0` is valid but uninformative, not a build or API failure.
Certificate tightness depends on the calibration count, family size, observed
p-values, confidence level, and envelope method. Do not change those inputs
post hoc solely to obtain a more attractive bound.

Choose the envelope `method` before inspecting the bound. Supported values are
`"mc_thc"`, `"mc_hc"`, `"mc_ks"`, `"ks"`, and `"mc_bj"`.

Method-specific keywords default to `None` and are resolved centrally:

| Method | Applicable options and effective defaults |
| --- | --- |
| `mc_thc` | `n_resamples=1000`, `seed=None`, `lower=0.01`, `upper=0.99`, `beta=0.5` |
| `mc_hc`, `mc_ks` | `n_resamples=1000`, `seed=None` |
| `mc_bj` | `n_resamples=1000`, `seed=None`, `precision=1e-8` |
| `ks` | No method-specific options; deterministic |

All methods accept `confidence=0.95` and `boost=True`. THC requires
`0 < beta <= 1`; unsupported values are rejected before Monte Carlo sampling.
Non-`None` inapplicable options raise an error. Method aliases such as `MC-THC`
remain accepted. The
certificate's seed controls Monte Carlo draws only; it does not inherit or alter
the fitting seed. `seed=None` draws fresh randomness at construction, and repeated
queries reuse those draws. Prepared queries use additional linear memory to avoid
repeated sorting and quadratic scans.

### Migration for 2.0 certificate restrictions

The stricter certificate checks are intentional behavioral breaks targeting
**2.0**, not a patch or minor release. Previously, certification accepted THC
`beta > 1` and randomized calibration/test score ties without a supported
justification. These configurations now raise errors. Supported certificate
numerics, empirical p-value computation, and ordinary selection are unchanged.

- For THC, use `0 < beta <= 1` (default `0.5`) and fix it before inspecting the
  curve; the implemented computational argument covers this range.
- For discrete or tied scoring rules, configure `Empirical(tie_break="classical")`
  before computing p-values. Randomized certification requires finite stored
  test scores and no exact calibration/test score ties. Do not edit scores or
  provenance on an existing snapshot to obtain certification.
- Recompute native p-values when snapshots lack recognized tie-mode provenance.
  External arrays still use `FDPCertificate.from_p_values(...)`, with the caller
  responsible for establishing the supported score construction.

This integration does not bump the package version or publish a release.

### Migration from the previous FDP API

This is an intentional clean break for FDP certification. Replace
`conformal_fdp_upper_bound_from_result(result, ...)` with
`result.fdp_bounds(...)`, or use `detector.fdp_bounds(x, ...)` directly. Replace
`conformal_fdp_upper_bound(p_values, ...)` with
`FDPCertificate.from_p_values(p_values, ...)`. `FDPCertificate` replaces
`FDPBoundResult`; the old names have been removed without aliases.

Move construction-time `thresholds=` to `certificate.to_frame(thresholds=...)`
or `bound_at(...)`. Omit unused method-specific options (in particular `seed`
and `n_resamples` for `ks`). Native entry points now reject unknown provenance;
use the expert route for external p-values. Certificate state is read-only.
Other selection and e-value APIs retain their behavior.

The only numerical correction concerns HC with a positive infinite Monte Carlo
cutoff: endpoint arithmetic previously could produce `NaN`; it now yields the
conservative unit envelope, so nonempty selections have bound 1 and empty ones
have bound 0. Finite-cutoff formulas, Monte Carlo sampling, and the quantile
convention are unchanged.

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
| Aggregate repeated split-conformal evidence | `DerandomizedSplits` with `select(...)` and e-BH |
| Explore p-value thresholds with a simultaneous realized-FDP certificate | `detector.fdp_bounds(x, ...)` |
| Maximize threshold discoveries below a certified FDP target | `certificate.threshold_for(max_fdp=...)`, then `select(...)` |
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
  develops simultaneous FDP bounds over conformal-p-value thresholds
  (arXiv preprint, Theorem 5.4 and Proposition 5.5).
- [Goeman and Solari (2011)](https://doi.org/10.1214/11-STS356),
  *Multiple Testing for Exploratory Research*, develops simultaneous confidence
  statements that remain valid after selecting a rejection set post hoc.
- [Javanmard and Montanari (2018)](https://doi.org/10.1214/17-AOS1629)
  develops LORD-style online FDR control.
