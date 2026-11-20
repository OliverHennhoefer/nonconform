<h1 align="center">
  <a href="https://oliverhennhoefer.github.io/nonconform/">
    <img src="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/source/assets/banner.png" alt="nonconform" width="900">
  </a>
</h1>

<p align="center">
  <strong>Anomaly detectors return scores. <code>nonconform</code> decides which to flag and keeps the share of false alarms below a level you set.</strong>
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
  <a href="https://oliverhennhoefer.github.io/nonconform/api/">API reference</a> ·
  <a href="https://arxiv.org/abs/2605.13642">Paper</a>
</p>

---

A detector ranks points by how unusual they look but leaves the cutoff to you.
The usual default, flagging the top 10% of scores, is a guess about how many
anomalies there are. `nonconform` calibrates the scores of your existing detector
against normal data, so the threshold follows from the error rate you can accept.

Below, the same Isolation Forest runs on seven [benchmark datasets](https://pypi.org/project/oddball/)
(20 random splits each), thresholded once by the usual top-10% default and once
by `nonconform`. The baseline trains on every row; `nonconform` holds some back
for calibration.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/readme/flags-dark.svg">
  <img src="https://raw.githubusercontent.com/OliverHennhoefer/nonconform/main/docs/img/readme/flags-light.svg" alt="True anomalies and false alarms flagged on seven datasets: the top-10% threshold flags about twice the real anomalies with 47 to 58 percent false; nonconform flags fewer, with 4 to 9 percent false" width="820">
</picture>

<details>
<summary>Flag counts per dataset</summary>

| Dataset | Anomalies | Top-10% threshold | `nonconform`, `alpha=0.1` |
| --- | ---: | ---: | ---: |
| Shuttle | 100 | 188 flagged, 47% false | 105 flagged, 8% false |
| HTTP | 100 | 190 flagged, 47% false | 99 flagged, 8% false |
| Satimage-2 | 71 | 157 flagged, 56% false | 70 flagged, 9% false |
| Thyroid | 61 | 116 flagged, 48% false | 42 flagged, 7% false |
| Fraud | 100 | 178 flagged, 50% false | 45 flagged, 9% false |
| Mammography | 100 | 153 flagged, 58% false | 6 flagged, 5% false |
| Cardio | 27 | 49 flagged, 53% false | 3 flagged, 4% false |

</details>

The average false share stays below the 10% target on every dataset. Where the detector
barely separates anomalies from normal data (Mammography, Cardio), `nonconform`
flags little rather than guess. Reproduce with
[`examples/benchmarks/default_thresholds.py`](examples/benchmarks/default_thresholds.py).

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

`nonconform` requires Python 3.11 or newer. Both discovery control and
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

Wrap a detector, fit it on normal data, and select anomalies at a target false
discovery rate (FDR). The example loads a benchmark dataset, so it needs the
`data` extra:

```python
from oddball import Dataset, load
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split

# Normal-only training data; a labelled test batch with 10% anomalies
x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=0)

alpha = 0.1  # target FDR: on average at most 10% of flags are false
detector = ConformalDetector(
    detector=IsolationForest(random_state=0),
    strategy=Split(n_calib=int(len(x_test) / alpha)),  # see the sizing rule below
    seed=0,
).fit(x_train)

flagged = detector.select(x_test, alpha=alpha)  # Boolean mask
hits = int(y_test[flagged].sum())
print(f"Flagged {flagged.sum()} of {len(x_test)}, {hits} true anomalies")
# Flagged 105 of 1000, 95 true anomalies
```

The conformal p-values behind the mask stay available in `detector.last_result`.
Any PyOD model, a recognized scikit-learn estimator, or a
[custom detector](https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/)
works in place of `IsolationForest`.

> [!TIP]
> **Calibration size.** With `n` calibration scores the smallest p-value is
> `1 / (n + 1)`, so flagging a lone anomaly in a batch of `m` needs roughly
> `n_calib >= m / alpha`. Below that, `select` can return nothing. More
> calibration is necessary but not sufficient: if the detector separates
> anomalies poorly, set a score threshold with `fpr_bounds` instead (below).

### Set a score threshold

If the requirement is "flag no more than 5% of normal points", certify a raw-score
threshold instead of targeting an FDR:

```python
certificate = detector.fpr_bounds(confidence=0.95)
selected = certificate.select(detector.score_samples(x_test), target_fpr=0.05)
print(f"Certified threshold: {certificate.threshold_for(0.05)}")
```

With 95% probability over the calibration draw, the threshold flags at most 5%
of future normal points. Calibration and future normal scores must be i.i.d.;
targets below the certificate's band floor (about 3.85% for 1,000 calibration scores at 95% confidence) give
an infinite threshold. See
[raw-score false-alarm control](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/#raw-score-false-alarm-control).

### Monitor a stream

A fitted `Split` detector can start a sequential monitor that raises an alarm when
the data stops looking like the calibration set.

<details>
<summary><strong>Monitor a stream for change</strong></summary>

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
`1 / alpha` bounds the probability of ever raising a false alarm on one stream.
See the [sequential monitoring guide](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/).

## Choose a workflow

| Goal | Start with |
| --- | --- |
| Calibrate and select anomalies in a batch | [`Split` and `select(...)`](https://oliverhennhoefer.github.io/nonconform/quickstart/) |
| Set a per-inlier raw-score false-alarm target | [`fpr_bounds(...)` and FPR certificates](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/#raw-score-false-alarm-control) |
| Aggregate evidence across random splits | [`DerandomizedSplits` and e-BH](https://oliverhennhoefer.github.io/nonconform/examples/derandomized_e_values/) |
| Monitor a stream for change | [Exchangeability martingales](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/) |
| Reuse more data for fitting and calibration | [`CrossValidation` or `JackknifeBootstrap`](https://oliverhennhoefer.github.io/nonconform/user_guide/conformalization_strategies/) |
| Account for covariate shift | [Weighted conformal inference](https://oliverhennhoefer.github.io/nonconform/user_guide/weighted_conformal/) |
| Calibrate groups with different normal behavior | [Group-conditional (Mondrian) p-values](https://oliverhennhoefer.github.io/nonconform/user_guide/conformal_inference/#group-conditional-mondrian-p-values) |
| Certify realized FDP across p-value thresholds | [`fdp_bounds(...)` and FDP certificates](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/#post-hoc-simultaneous-fdp-bounds) |
| Bring a custom or third-party detector | [Detector compatibility](https://oliverhennhoefer.github.io/nonconform/user_guide/detector_compatibility/) |

## Coding agents

An [agent skill](https://github.com/OliverHennhoefer/nonconform/blob/main/plugins/nonconform/skills/nonconform/SKILL.md)
teaches coding agents to use `nonconform` without overclaiming. Install it in
Claude Code (`/plugin marketplace add OliverHennhoefer/nonconform`, then
`/plugin install nonconform@nonconform`) or Codex
(`codex plugin marketplace add OliverHennhoefer/nonconform`, then
`codex plugin add nonconform@nonconform`).

## Statistical scope

> [!IMPORTANT]
> Every guarantee rests on assumptions. Conformal p-values need calibration data
> and normal test points to be exchangeable; FDR claims need the assumptions of
> the chosen multiple-testing procedure; FPR certificates need i.i.d. scores from
> an independently fitted detector; weighted workflows need a credible
> covariate-shift model; Ville alarms cover one stream, while CUSUM and
> Shiryaev-Roberts thresholds need separate calibration.

`nonconform` calibrates detector scores; it cannot make an unsuitable detector
or a mismatched calibration set valid. Handle spatial or temporal dependence
before relying on exchangeability. Read the guides to
[FDR control](https://oliverhennhoefer.github.io/nonconform/user_guide/fdr_control/)
and [sequential monitoring](https://oliverhennhoefer.github.io/nonconform/user_guide/exchangeability_martingales/)
before applying a guarantee to a new problem.

## Citation

If you use `nonconform` in academic work, please cite the
[accompanying paper](https://arxiv.org/abs/2605.13642):

```bibtex
@inproceedings{Hennhoefer2026,
author = {Hennhöfer, Oliver and Kirsch, Maximilian and Preisach, Christine},
booktitle = {Proceedings of the Fifteenth Symposium on Conformal and Probabilistic Prediction with Applications},
pages = {613--632},
publisher = {PMLR},
title = {{Conformal Anomaly Detection in Python: Moving Beyond Heuristic Thresholds with nonconform}},
url = {https://proceedings.mlr.press/v329/hennhofer26a.html},
year = {2026}
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

<p align="center">
  <a href="https://www.dlr.de/">
    <img src="https://www.dlr.de/de/pt-lf/aktuelles/pressematerial/logos/bmwk/vorschaubild_bmwk_logo-mit-foerderzusatz_en/@@images/image-600-ea91cd9090327104991124b30fe1de7d.png" alt="Funding acknowledgement" width="250">
  </a>
</p>
