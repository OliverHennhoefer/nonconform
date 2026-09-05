---
description: "Independently runnable nonconform examples for batch, conditional, resampling, weighted, derandomized e-value, FDR, and sequential workflows."
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
| [Data-efficient resampling](resampling_conformal.md) | A fixed holdout is costly and resampling is justified | Strategy comparison with fit cost, FDP, and power |
| [Weighted conformal](weighted_conformal.md) | The target null follows a defensible covariate-shift model | Weighted p-values, WCS mask, and weight diagnostics |
| [Derandomized conformal e-values](derandomized_e_values.md) | Random calibration splits make selections unstable | `DerandomizedSplits` with `detector.select()`, e-values, and e-BH mask |
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

!!! note "Examples measure behavior; guides state scope"

    A successful run shows that the API path executes on the displayed data.
    It does not verify exchangeability, density-ratio correctness, dependence
    conditions, or deployment performance. Follow each example's links to the
    corresponding user guide before making a statistical claim.
