# Examples

Practical examples demonstrating different conformal anomaly detection approaches.

## Getting Started

| Example | Difficulty | What You'll Learn |
|---------|------------|-------------------|
| [Classical Conformal](classical_conformal.md) | Beginner | Basic split conformal detection, FDR control, p-value interpretation |
| [Data-Efficient Resampling](resampling_conformal.md) | Intermediate | CV+, Jackknife+, and JaB+ for small-data calibration |

## Advanced Strategies

| Example | Difficulty | What You'll Learn |
|---------|------------|-------------------|
| [Conditional Conformal](conditional_conformal.md) | Intermediate | Conditionally calibrated conformal p-values with BH-style FDR selection |

## Special Topics

| Example | Difficulty | What You'll Learn |
|---------|------------|-------------------|
| [Weighted Conformal](weighted_conformal.md) | Advanced | Handling distribution shift between training and test data |
| [FDR Control](fdr_control.md) | Intermediate | Multiple testing correction, Benjamini-Hochberg procedure |

## What Each Example Covers

**[Classical Conformal](classical_conformal.md)** - Start here if you're new to nonconform. Learn the core workflow: wrap a detector, compute p-values, and apply FDR control. Includes visualization of results.

**[Data-Efficient Resampling](resampling_conformal.md)** - Learn CV+, Jackknife+, and JaB+ together as one family of strategies for using scarce training data efficiently. Start here when a fixed calibration holdout would cost too much power.

**[Conditional Conformal](conditional_conformal.md)** - Apply conditional calibration maps (`mc`, `simes`, `dkwm`, `asymptotic`) to empirical conformal p-values, then run BH-style FDR selection.

**[Weighted Conformal](weighted_conformal.md)** - Handle covariate shift scenarios where your test data comes from a different distribution than your training data. Essential for real-world deployment.

**[FDR Control](fdr_control.md)** - Deep dive into False Discovery Rate control. Understand when to use BH vs weighted methods and how to evaluate FDR performance.

## Prerequisites

All examples assume you have installed nonconform with the PyOD and data extras:

```bash
pip install "nonconform[pyod,data]"
```

Most examples use the PyOD `LOF` detector and benchmark datasets from `oddball`. Each example is self-contained and can be run independently.
