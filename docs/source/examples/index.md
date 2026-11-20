---
description: "Independently runnable nonconform examples for batch, conditional, group-conditional, resampling, weighted, derandomized e-value, FDR, and sequential workflows."
---

# Examples

Every Python block in this section is self-contained: it includes its imports,
data construction or loading, fitting, inference, and output. Copy any block
into a fresh Python process after installing the dependencies named on its page.

## Choose an example

| Example | Start here when | Main output |
|---|---|---|
| [Classical split conformal](classical_conformal.md) | Calibration and test nulls are exchangeable | Empirical p-values and BH discovery mask |
| [Conditional conformal](conditional_conformal.md) | You need calibration-set-conditional p-value maps | Conditionally transformed p-values and BH mask |
| [Group-conditional conformal](group_conditional.md) | Groups differ in normal behavior or the test group mix shifts | Pooled and per-group (Mondrian) BH selections compared |
| [Data-efficient resampling](resampling_conformal.md) | A fixed holdout is costly and resampling is justified | Strategy comparison with fit cost, FDP, and power |
| [Weighted conformal](weighted_conformal.md) | The target null follows a defensible covariate-shift model | Weighted p-values, WCS mask, and weight diagnostics |
| [Derandomized conformal e-values](derandomized_e_values.md) | Random calibration splits make selections unstable | Uniformly averaged split evidence and e-BH mask |
| [FDR control and FDP bounds](fdr_control.md) | You need to compare multiple-testing targets | Pointwise, BH, BY, and simultaneous FDP certificate |
| [Exchangeability martingales](../user_guide/exchangeability_martingales.md#basic-sequential-usage) | You monitor an ordered stream for change | Sequential p-values, martingale evidence, and alarms |

## Dependency guide

The synthetic scikit-learn examples need only the core installation:

```bash
pip install nonconform
```

The classical benchmark and derandomized e-value examples also use PyOD and
oddball:

```bash
pip install "nonconform[data,pyod]"
```

No example assumes variables created by a previous code block.

## Notebooks

The notebooks in
[`examples/`](https://github.com/OliverHennhoefer/nonconform/tree/main/examples)
open in Colab. The first three have plots, accompany the
[use cases](../use_cases/index.md), and install what they need. For the others,
run `!pip install "nonconform[data,pyod]"` in a first cell.

| Notebook | Run |
|---|---|
| Screen a batch of network traffic with `select` | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/batch_screening.ipynb) |
| Replace `contamination` with a certified false-alarm rate | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/false_alarm_threshold.ipynb) |
| Monitor a sensor stream with a martingale | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/sensor_stream.ipynb) |
| Certify a raw-score false-alarm rate | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/fpr_bounds.ipynb) |
| Bound the false discovery proportion of one batch | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/fdp_bounds.ipynb) |
| Storey adaptive BH | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/adaptive_bh.ipynb) |
| Conditionally calibrated p-values | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/conditional_calibration.ipynb) |
| Calibrate on data separate from training | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/detached_calibration.ipynb) |
| Compare p-value estimators | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/estimation_comparison.ipynb) |
| Group-conditional p-values | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/group_conditional.ipynb) |
| Derandomized e-values | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/derandomized_e_values.ipynb) |
| Exchangeability martingale on fraud data | [Open in Colab](https://colab.research.google.com/github/OliverHennhoefer/nonconform/blob/main/examples/exchangeability_martingale.ipynb) |

A successful run shows that the code executes on the displayed data. It does
not verify the assumptions behind a statistical claim; see
[Assumptions and guarantees](../guarantees.md).
