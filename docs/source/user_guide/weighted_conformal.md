---
description: "Use weighted conformal p-values and weighted conformalized selection under a documented covariate-shift model."
---

# Weighted conformal inference

Weighted conformal inference addresses a specific departure from standard
exchangeability: the covariate distribution of target observations differs
from that of calibration observations, while the relevant conditional
mechanism remains stable.

It is not a general-purpose drift correction. Use it only when the shift model
is defensible.

!!! abstract "Workflow in one paragraph"

    Add a `weight_estimator` to `ConformalDetector`, fit the detector on
    reference data, and pass the complete target family to `select(...)`. The
    weight estimator distinguishes calibration covariates from target
    covariates and estimates a density ratio. `nonconform` then computes
    weighted conformal p-values and applies weighted conformalized selection
    (WCS). Inspect the stored weights and validate the shift model before
    relying on the result.

## Assumptions

Let `P_X` denote the calibration covariate distribution and `Q_X` the target
covariate distribution. Weighted conformal methods use an importance ratio

$$
w(x)=\frac{dQ_X}{dP_X}(x).
$$

The core requirements are:

1. **Appropriate shift model.** The feature distribution may change, but the
   null/anomaly semantics and relevant conditional mechanism must remain
   stable in the sense required by the chosen weighted-conformal result.
2. **Support overlap.** Target observations must lie in regions where the
   calibration distribution has support. Finite weighting cannot recover a
   missing reference population.
3. **Valid score construction.** The fitted detector and learned preprocessing
   must not use the calibration or target outcomes being tested.
4. **Suitable weights.** Exact results assume known likelihood ratios; using
   estimated ratios adds modeling error that must be assessed.
5. **Matching selection procedure.** Weighted conformal p-values generally
   cannot simply be passed to BH. Use WCS for the finite-sample FDR result
   developed for this setting.

Examples that may fit the model include a change in sensor mix, geography, or
sampling policy when the definition and mechanism of a normal case remain
stable. A changed anomaly mechanism, new measurement semantics, label shift,
or missing target support is not repaired merely by turning on weights.

## Complete weighted selection

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(2_000, 3))
x_target = np.vstack(
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

selected = detector.select(x_target, alpha=0.1)
result = detector.last_result
assert result is not None
assert result.p_values is not None
assert result.calib_weights is not None
assert result.test_weights is not None

print(f"discoveries: {selected.sum()}")
print("smallest p-values:", np.sort(result.p_values)[:5])
print("calibration weight range:", np.ptp(result.calib_weights))
print("target weight range:", np.ptp(result.test_weights))
```

In weighted mode, `select(...)` performs WCS. Calling SciPy BH on the stored
weighted p-values would be a different procedure with a different, generally
unsupported dependence argument.

## What the package fits

With `Split` and weighted mode:

1. the strategy splits reference rows into proper training and calibration;
2. the anomaly detector is fitted on proper training rows;
3. calibration rows are scored by that fixed detector;
4. when a target batch is supplied, the weight estimator is fitted to
   distinguish calibration covariates (class `0`) from target covariates
   (class `1`);
5. probability odds estimate the target-to-calibration density ratio;
6. weighted p-values and WCS selection are computed for that target batch.

The logistic factory uses a standardization pipeline and balanced logistic
regression. The forest factory uses a balanced random forest. Both default to
quantile clipping at `clip_quantile=0.05` across the combined calibration and
target weights.

!!! warning "Clipping changes the estimand"

    Clipping can stabilize numerical behavior and reduce the influence of
    extreme estimated ratios, but it replaces the original ratio with a
    truncated one. It is a bias-variance and robustness choice, not a proof of
    overlap. Report it and assess sensitivity to it.

## Weighted p-values

For calibration scores `S_i`, calibration weights `w_i`, test score `S(x)`, and
test weight `w(x)`, classical weighted empirical mode computes

$$
p(x)=\frac{w(x)+\sum_{i=1}^{n_{\mathrm{cal}}}
w_i\mathbf{1}\{S_i\ge S(x)\}}
{w(x)+\sum_{i=1}^{n_{\mathrm{cal}}}w_i}.
$$

This deterministic formula includes tied calibration mass and the test point's
own mass. With unit weights, it reduces to the classical unweighted empirical
formula.

With `Empirical(tie_break="randomized")`, `nonconform` uses

$$
p(x)=\frac{\sum_i w_i\mathbf{1}\{S_i>S(x)\}
+U\left(w(x)+\sum_i w_i\mathbf{1}\{S_i=S(x)\}\right)}
{w(x)+\sum_i w_i},
$$

where `U` is uniform on `[0, 1]`. This removes the positive discrete floor by
randomizing both tied mass and the test point's mass. Supply `seed` for
reproducibility.

## Weight-estimator choices

### Logistic density-ratio estimator

Use the logistic factory as a transparent baseline when calibration-to-target
separation is adequately represented by a regularized linear decision boundary
after standardization.

```python
from nonconform import logistic_weight_estimator

weight_estimator = logistic_weight_estimator(
    regularization=1.0,
    clip_quantile=0.05,
    class_weight="balanced",
    max_iter=1000,
)

print(type(weight_estimator.base_estimator).__name__)
print(weight_estimator.base_estimator.get_params()["logisticregression__C"])
print(weight_estimator.clip_quantile)
```

`regularization` is passed as scikit-learn logistic regression's `C`; larger
values mean weaker L2 regularization. `"auto"` uses `C=1.0`.

### Forest density-ratio estimator

Use the forest factory when nonlinear separation is plausible and validate its
probability estimates carefully. Flexible discrimination can overfit,
especially with limited target data.

```python
from nonconform import forest_weight_estimator

weight_estimator = forest_weight_estimator(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=10,
    clip_quantile=0.05,
)

print(type(weight_estimator.base_estimator).__name__)
print(weight_estimator.base_estimator.n_estimators)
print(weight_estimator.clip_quantile)
```

There is no generally superior estimator. Compare held-out domain
discrimination, weight stability, clipping sensitivity, and downstream
behavior under realistic shifted evaluation designs.

### Bootstrap-bagged weights

`BootstrapBaggedWeightEstimator` repeatedly fits a base weight estimator to
balanced bootstrap samples, scores every original calibration and target row,
and aggregates log weights by a geometric mean. Its current
`scoring_mode="frozen"` only serves the exact batches used during fitting.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.weighting import BootstrapBaggedWeightEstimator

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(400, 3))
x_target = rng.normal(loc=0.5, size=(30, 3))

bagged_weights = BootstrapBaggedWeightEstimator(
    base_estimator=logistic_weight_estimator(),
    n_bootstraps=5,
    clip_quantile=0.05,
)
detector = ConformalDetector(
    detector=IsolationForest(n_estimators=30, random_state=42),
    strategy=Split(n_calib=0.3),
    weight_estimator=bagged_weights,
    seed=42,
).fit(x_reference)

p_values = detector.compute_p_values(x_target)
print(p_values[:5])
```

Bagging adds fitting cost and is not automatically more accurate. Increase the
bootstrap count only when repeated evaluation shows that the added stability
is worth the cost.

## Batch-specific state

By default, every `compute_p_values(...)` or `select(...)` call refits the weight
estimator for its input batch. This is intentional: the batch represents the
target distribution used in density-ratio estimation.

For an explicit state transition, prepare weights for the exact batch:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(600, 3))
x_target = rng.normal(loc=0.5, size=(40, 3))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
).fit(x_reference)

detector.prepare_weights_for(x_target)
p_values = detector.compute_p_values(x_target, refit_weights=False)
print(p_values[:5])
```

The default content check rejects a same-sized but different batch. Disabling
`verify_prepared_batch_content` removes the digest check and leaves identity
enforcement to the caller.

Arbitrarily splitting one target family into chunks changes the fitted density
ratio, the weighted p-values, and the WCS problem. Do not describe chunking as
an equivalent memory optimization.

## WCS pruning

`select(...)` accepts `Pruning.DETERMINISTIC`, `Pruning.HOMOGENEOUS`, and
`Pruning.HETEROGENEOUS` from `nonconform.fdr` or `nonconform.enums`.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator
from nonconform.fdr import Pruning

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(2_000, 2))
x_target = np.vstack(
    [rng.normal(loc=0.5, size=(18, 2)), rng.normal(loc=6.0, size=(2, 2))]
)

detector = ConformalDetector(
    detector=IsolationForest(n_estimators=50, random_state=42),
    strategy=Split(n_calib=0.4),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
).fit(x_reference)

selected = detector.select(
    x_target,
    alpha=0.1,
    pruning=Pruning.HETEROGENEOUS,
    seed=7,
)
print(np.flatnonzero(selected))
```

Deterministic pruning is the default. Homogeneous pruning uses one shared
uniform random variable; heterogeneous pruning uses independent uniform
variables. A selection seed makes randomized pruning reproducible.

## Diagnostics that earn attention

After `compute_p_values(...)` or `select(...)`, inspect `last_result`. Weight
quantiles and an importance-weight effective sample size (ESS) are useful
descriptive summaries:

$$
\operatorname{ESS}(w)=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split, logistic_weight_estimator

rng = np.random.default_rng(42)
x_reference = rng.normal(size=(700, 3))
x_target = rng.normal(loc=0.8, size=(80, 3))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    weight_estimator=logistic_weight_estimator(),
    seed=42,
).fit(x_reference)
detector.compute_p_values(x_target)

result = detector.last_result
assert result is not None
assert result.calib_weights is not None
assert result.test_weights is not None

def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())

for name, weights in {
    "calibration": result.calib_weights,
    "target": result.test_weights,
}.items():
    print(
        name,
        {
            "quantiles": np.quantile(weights, [0.0, 0.05, 0.5, 0.95, 1.0]),
            "ess": effective_sample_size(weights),
            "count": len(weights),
        },
    )
```

ESS is a diagnostic, not a validity theorem or a universal pass/fail test. A
small ESS relative to the number of rows signals weight concentration and weak
effective support, but no task-independent cutoff separates safe from unsafe.

Also assess:

- out-of-sample calibration of the domain classifier;
- whether calibration and target samples are nearly perfectly separable;
- stability across seeds and plausible estimator specifications;
- sensitivity to `clip_quantile`;
- empirical p-value calibration and FDP on untouched shifted null data;
- whether target observations occupy regions absent from calibration data.

Marginal two-sample tests can flag distribution differences but cannot prove
the joint covariate-shift condition or support overlap.

## Strategy and estimator scope

`Split` is the recommended baseline for weighted mode because proper training,
calibration, and target roles are explicit. Resampling strategies expose
calibration samples too, but their score aggregation and dependence differ.
Do not transfer the split weighted-conformal guarantee to those combinations
without a matching argument.

`ConditionalEmpirical` intentionally rejects weighted mode.
`Probabilistic` can numerically consume weights, but its KDE p-values are
model-based rather than the exact discrete empirical construction. State that
distinction and validate it separately.

`ExchangeabilityMonitor.from_split_detector(...)` rejects weighted detectors.
Batch-specific density-ratio refitting is not the frozen sequential rank
construction required for the monitor's martingale guarantee.

## Common failure modes

| Symptom | Likely issue | Action |
|---|---|---|
| Nearly perfect calibration/target discrimination | Poor overlap or easily separable domains | Revisit support and shift assumptions; collect reference data covering the target |
| Highly concentrated weights | Limited effective support, classifier instability, or extrapolation | Inspect features, compare estimators, and report clipping sensitivity |
| Large changes across target batch definitions | Batch-specific ratio estimation | Define the target family before fitting weights and avoid arbitrary chunking |
| Weighted and unweighted results differ sharply | Material estimated shift, poor weights, or both | Validate on shifted labeled/null data rather than choosing the preferred result |
| No discoveries | Coarse weighted p-values, conservative pruning, weak detector, or little signal | Inspect scores, p-values, weights, and WCS rejection sizes without weakening assumptions post hoc |

## Checklist

- State `P_X`, `Q_X`, and why covariate shift is plausible.
- Confirm target support is represented in calibration data.
- Keep detector fitting and learned preprocessing independent of calibration and
  target outcomes.
- Fit weights on the complete target family used by WCS.
- Inspect weight concentration, ESS, clipping, and estimator sensitivity.
- Use WCS through weighted `select(...)`, not ordinary BH.
- Treat randomized pruning and randomized tie handling as genuine
  randomization and record seeds.
- Report estimated-weight limitations with every validity claim.

## References

- [Tibshirani et al. (2019), *Conformal Prediction Under Covariate Shift*](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)
  develops weighted conformal prediction under covariate shift, including the
  likelihood-ratio requirement.
- [Jin and Candès (2023), *Model-free Selective Inference under Covariate Shift via Weighted Conformal p-values*](https://arxiv.org/abs/2307.09291)
  introduces weighted conformal p-values and WCS, including the outlier
  detection extension under inlier-distribution shift.
- [Sugiyama et al. (2012), *Density Ratio Estimation in Machine Learning*](https://doi.org/10.1017/CBO9781139035613)
  provides broader background on density-ratio estimation.
