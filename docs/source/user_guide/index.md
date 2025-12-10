# User Guide

This guide covers everything you need to know to use nonconform effectively, from the underlying theory to production deployment.

## Getting Started

If you're new to conformal prediction, start here:

| Page | Description |
|------|-------------|
| [Statistical Concepts](statistical_concepts.md) | Quick reference for key statistical terms (p-values, FDR, exchangeability) |
| [Conformal Inference](conformal_inference.md) | Deep dive into how conformal prediction works and why it provides guarantees |

## Core Concepts

| Page | Description |
|------|-------------|
| [Conformalization Strategies](conformalization_strategies.md) | Split, Cross-Validation, Jackknife+, and Bootstrap strategies explained |
| [Choosing Strategies](choosing_strategies.md) | Decision framework: which strategy to use for your dataset and requirements |
| [Detector Compatibility](detector_compatibility.md) | How to use PyOD, scikit-learn, or your own custom detectors |

## Advanced Topics

| Page | Description |
|------|-------------|
| [FDR Control](fdr_control.md) | Control false discovery rates when testing many observations |
| [Weighted Conformal](weighted_conformal.md) | Handle distribution shift between training and test data |

## Practical Usage

| Page | Description |
|------|-------------|
| [Input Validation](input_validation.md) | Parameter constraints and what error messages mean |
| [Batch Evaluation](batch_evaluation.md) | Evaluate performance on labeled test sets |
| [Streaming Evaluation](streaming_evaluation.md) | Online evaluation for real-time detection |
| [Best Practices](best_practices.md) | Production patterns, data preparation, and model selection |
| [Logging](logging.md) | Configure progress bars and debug output |
| [Troubleshooting](troubleshooting.md) | Solutions to common issues and migration from older versions |

## Recommended Reading Order

1. **New to conformal prediction?** Start with [Statistical Concepts](statistical_concepts.md), then [Conformal Inference](conformal_inference.md)
2. **Choosing a strategy?** Read [Choosing Strategies](choosing_strategies.md)
3. **Going to production?** Review [Best Practices](best_practices.md) and [Troubleshooting](troubleshooting.md)
4. **Dealing with distribution shift?** Study [Weighted Conformal](weighted_conformal.md)
