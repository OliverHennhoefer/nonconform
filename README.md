<h1 align="center">
  <a href="https://oliverhennhoefer.github.io/nonconform/">
    <img src="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/source/assets/banner.png" alt="nonconform" width="900">
  </a>
</h1>

<p align="center">
  <strong>Calibrate scores. Control discoveries. Monitor change.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/nonconform/"><img src="https://img.shields.io/pypi/v/nonconform.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/nonconform/"><img src="https://img.shields.io/pypi/pyversions/nonconform.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/OliverHennhoefer/nonconform/actions/workflows/codecov.yml"><img src="https://img.shields.io/github/actions/workflow/status/OliverHennhoefer/nonconform/codecov.yml?branch=main&label=Tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/OliverHennhoefer/nonconform"><img src="https://codecov.io/gh/OliverHennhoefer/nonconform/branch/main/graph/badge.svg?token=Z78HU3I26P" alt="Code coverage"></a>
  <a href="https://oliverhennhoefer.github.io/nonconform/"><img src="https://img.shields.io/github/actions/workflow/status/OliverHennhoefer/nonconform/docs.yml?branch=main&label=Documentation" alt="Documentation"></a>
  <a href="https://github.com/OliverHennhoefer/nonconform/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/nonconform.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://oliverhennhoefer.github.io/nonconform/">Documentation</a> ·
  <a href="https://oliverhennhoefer.github.io/nonconform/quickstart/">Batch workflow</a> ·
  <a href="https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/">Sequential workflow</a> ·
  <a href="https://oliverhennhoefer.github.io/nonconform/api/">API reference</a> ·
  <a href="https://arxiv.org/abs/2605.13642">Paper</a>
</p>

`nonconform` turns anomaly scores into conformal evidence for two primary
workflows: batch discovery control and sequential change monitoring. Wrap a
supported scikit-learn estimator, a [PyOD](https://pyod.readthedocs.io/) model,
or a custom detector:

- **Batch:** Use calibrated p-values directly or call `select(...)` to apply
  false discovery rate (FDR) control.
- **Stream:** Use conformal martingales to accumulate evidence against
  exchangeability and trigger configured alarms.

## Why nonconform?

- **Calibrate anomaly scores** into conformal p-values using reference data.
- **Control batch discoveries** with `ConformalDetector.select(...)`, which
  combines calibration and FDR control in one workflow.
- **Combine repeated split evidence** with `DerandomizedSplits`, automatic
  model fitting, and e-value-based selection through the same `select(...)` API.
- **Monitor streams for change** with conformal martingales, anytime evidence
  against exchangeability, and configurable alarms.
- **Keep your detector** through support for PyOD, recognized scikit-learn
  estimators, and protocol-compliant custom models.
- **Adapt the calibration** with split, cross-validation, and
  jackknife+-after-bootstrap strategies.
- **Handle advanced settings** with weighted conformal methods and post-hoc FDP
  bounds.

<p align="center">
  <strong>Works with</strong>
  &nbsp;&nbsp;&nbsp;
  <a href="https://scikit-learn.org/"><img src="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/integrations/scikit-learn.svg?v=dc048ee" alt="scikit-learn" height="44" align="middle"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://pyod.readthedocs.io/"><img src="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/integrations/pyod.svg?v=1" alt="PyOD" height="44" align="middle"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/"><img src="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/integrations/custom.svg?v=1" alt="Custom AnomalyDetector protocol" height="44" align="middle"></a>
</p>

## Installation

`nonconform` requires Python 3.12 or newer. Both batch discovery control and
sequential monitoring are included in the core installation.

```bash
pip install nonconform
```

For the PyOD detector collection and benchmark datasets:

```bash
pip install "nonconform[pyod,data]"
```

<details>
<summary><strong>Optional extras</strong></summary>

| Extra | Adds |
| --- | --- |
| `pyod` | PyOD anomaly detectors |
| `data` | `oddball` benchmark datasets and PyArrow support |
| `fdr` | Streaming FDR procedures from `online-fdr` |
| `probabilistic` | KDE-based probabilistic estimation and tuning |
| `all` | Every optional feature |

</details>

## Quick start

### Batch discovery control

This core-only example demonstrates the batch lane. The detector is trained on
normal data, part of which is reserved automatically for conformal calibration.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

rng = np.random.default_rng(42)
x_train = rng.normal(size=(1_000, 2))
x_test = np.vstack([
    rng.normal(size=(200, 2)),
    rng.normal(loc=5.0, size=(20, 2)),
])

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_train)

discoveries = detector.select(x_test, alpha=0.05)
p_values = detector.last_result.p_values

print(f"Selected {discoveries.sum()} of {len(x_test)} observations")
```

> [!NOTE]
> `discoveries` is a Boolean mask. Here, `alpha=0.05` is the target FDR level,
> not a per-observation score threshold. The underlying conformal p-values remain
> available through `last_result` for inspection or downstream analysis.

### Sequential change monitoring

A fitted `Split` detector can initialize the stream lane without refitting its
scoring model. The example is self-contained so it can be copied independently
of the batch example.

<details>
<summary><strong>Show sequential monitoring example</strong></summary>

```python
import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, SimpleJumperMartingale
from nonconform.monitoring import ExchangeabilityMonitor

rng = np.random.default_rng(42)
x_train = rng.normal(size=(1_000, 2))

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    seed=42,
).fit(x_train)

alpha = 0.05
monitor = ExchangeabilityMonitor.from_split_detector(
    detector,
    martingale=SimpleJumperMartingale(
        alarm_config=AlarmConfig(restarted_ville_threshold=1 / alpha)
    ),
    seed=42,
)

# Stable observations followed by a distribution shift
x_stream = np.vstack([
    rng.normal(size=(50, 2)),
    rng.normal(loc=3.0, size=(50, 2)),
])

for x_t in x_stream:
    state = monitor.update(x_t)
    if "restarted_ville" in state.triggered_alarms:
        print(f"Change alarm at step {state.evidence_step}")
        break
```

</details>

Under the sequential validity assumptions, the restarted Ville alarm at
`1 / alpha` controls the probability of ever crossing on one stream. It does
not control FDR across multiple streams. See the
[sequential monitoring guide](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/)
for the full guarantee scope and other alarm statistics.

## Choose a workflow

| Goal | Start with |
| --- | --- |
| Calibrate and select anomalies in a batch | [`Split` and `select(...)`](https://oliverhennhoefer.github.io/nonconform/quickstart/) |
| Aggregate evidence across random splits | [`DerandomizedSplits` and e-BH](https://oliverhennhoefer.github.io/nonconform/examples/derandomized_e_values/) |
| Monitor a stream for change | [Exchangeability martingales](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/) |
| Reuse more data for fitting and calibration | [`CrossValidation` or `JackknifeBootstrap`](https://oliverhennhoefer.github.io/nonconform/user_guide/conformalization_strategies/) |
| Account for covariate shift | [Weighted conformal inference](https://oliverhennhoefer.github.io/nonconform/user_guide/weighted_conformal/) |
| Certify a chosen p-value threshold post hoc | [FDP upper bounds](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/) |
| Bring a custom or third-party detector | [Detector compatibility](https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/) |

## Statistical scope

> [!IMPORTANT]
> **Guarantees are assumption-dependent.** Standard conformal workflows require
> calibration data and null test cases to be exchangeable. FDR claims additionally
> require valid p-values or the applicable aggregate null-evidence condition,
> together with the assumptions of the selected multiple-testing procedure.
> Weighted workflows require a plausible covariate-shift model, support
> overlap, and reliable weights. Sequential martingales require valid sequential
> conformal p-values; Ville thresholds provide false-alarm control for one valid
> stream, while CUSUM and Shiryaev-Roberts thresholds are change-evidence triggers
> that require separate calibration.

`nonconform` calibrates detector scores; it cannot make an unsuitable detector
or mismatched calibration set valid. Spatial or temporal dependence must be
handled explicitly before applying standard exchangeability-based claims. See
the guides to [FDR control](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/)
and [sequential monitoring](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/)
before relying on error-control statements in a new application.

## Citation

If you use `nonconform` in academic work, please cite the
[accompanying paper](https://arxiv.org/abs/2605.13642):

```bibtex
@misc{hennhoefer2026,
  title={Conformal Anomaly Detection in Python: Moving Beyond Heuristic Thresholds with 'nonconform'},
  author={Oliver Hennhöfer and Maximilian Kirsch and Christine Preisach},
  year={2026},
  eprint={2605.13642},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2605.13642},
}
```

## Project

Read the [documentation](https://oliverhennhoefer.github.io/nonconform/), browse
the [changelog](https://github.com/OliverHennhoefer/nonconform/blob/main/CHANGELOG.md),
or report a problem in the [issue tracker](https://github.com/OliverHennhoefer/nonconform/issues).
Contributions are welcome; start with the
[contributing guide](https://oliverhennhoefer.github.io/nonconform/contributing/).
`nonconform` is distributed under the
[BSD 3-Clause License](https://github.com/OliverHennhoefer/nonconform/blob/main/LICENSE).

---

<p align="center">
  <a href="https://www.dlr.de/">
    <img src="https://www.dlr.de/de/pt-lf/aktuelles/pressematerial/logos/bmwk/vorschaubild_bmwk_logo-mit-foerderzusatz_en/@@images/image-600-ea91cd9090327104991124b30fe1de7d.png" alt="Funding acknowledgement" width="250">
  </a>
</p>
