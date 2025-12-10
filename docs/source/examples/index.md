# Examples

Practical examples demonstrating different conformal anomaly detection approaches.

## Getting Started

| Example | Difficulty | What You'll Learn |
|---------|------------|-------------------|
| [Classical Conformal](classical_conformal.md) | Beginner | Basic split conformal detection, FDR control, p-value interpretation |
| [Cross-Validation Conformal](cross_val_conformal.md) | Beginner | K-fold cross-validation for better data efficiency |

## Advanced Strategies

| Example | Difficulty | What You'll Learn |
|---------|------------|-------------------|
| [Jackknife Conformal](jackknife_conformal.md) | Intermediate | Jackknife+ strategy for theoretical guarantees |
| [Bootstrap Conformal](bootstrap_conformal.md) | Intermediate | JaB+ strategy for uncertainty quantification |

## Special Topics

| Example | Difficulty | What You'll Learn |
|---------|------------|-------------------|
| [Weighted Conformal](weighted_conformal.md) | Advanced | Handling distribution shift between training and test data |
| [FDR Control](fdr_control.md) | Intermediate | Multiple testing correction, Benjamini-Hochberg procedure |

## What Each Example Covers

**[Classical Conformal](classical_conformal.md)** - Start here if you're new to nonconform. Learn the core workflow: wrap a detector, compute p-values, and apply FDR control. Includes visualization of results.

**[Cross-Validation Conformal](cross_val_conformal.md)** - Use all your training data for both fitting and calibration using k-fold CV. Good for smaller datasets where you can't afford to reserve data.

**[Jackknife Conformal](jackknife_conformal.md)** - Apply the Jackknife+ strategy, which provides finite-sample theoretical guarantees. A good balance between computational cost and accuracy.

**[Bootstrap Conformal](bootstrap_conformal.md)** - Use the Jackknife+-after-Bootstrap (JaB+) strategy for robust uncertainty quantification. Best when you need to understand detection stability.

**[Weighted Conformal](weighted_conformal.md)** - Handle covariate shift scenarios where your test data comes from a different distribution than your training data. Essential for real-world deployment.

**[FDR Control](fdr_control.md)** - Deep dive into False Discovery Rate control. Understand when to use BH vs weighted methods and how to evaluate FDR performance.

## Prerequisites

All examples assume you have installed nonconform with the PyOD and data extras:

```bash
pip install "nonconform[pyod,data]"
```

Most examples use the PyOD `LOF` detector and benchmark datasets from `oddball`. Each example is self-contained and can be run independently.
