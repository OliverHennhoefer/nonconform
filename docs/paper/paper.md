---
title: 'nonconform: Conformal Anomaly Detection (Python)'
tags:
  - Python
  - Anomaly detection
  - Conformal Inference
  - Conformal Anomaly Detection
  - Uncertainty Quantification
  - False Discovery Rate
authors:
  - name: Oliver Hennhöfer
    orcid: 0000-0001-9834-4685
    affiliation: 1
affiliations:
 - name: Intelligent Systems Research Group (ISRG), Karlsruhe University of Applied Sciences (HKA), Karlsruhe, Germany
   index: 1
date: 10 September 2025
bibliography: paper.bib
---

# Summary

The Python package `nonconform` provides statistically principled uncertainty quantification for unsupervised anomaly detection.
It implements methods from conformal anomaly detection [@Laxhammar2010; @Bates2023; @Jin2023] based on the principles of one-class classification [@Petsche1994].
The ability to quantify uncertainty is a fundamental requirement for AI systems in safety-critical domains, where reliable decision-making is essential.

Based on the underlying principles of conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012], `nonconform` converts raw anomaly scores from an underlying detection model into statistically valid $p$-values.
This is achieved by calibrating the model on a hold-out set of normal data; the $p$-value for a new test instance is then calculated as the relative rank of its anomaly score compared to the scores from the calibration set.
By framing anomaly detection as a series of statistical hypothesis tests, these $p$-values allow for the systematic control of the False Discovery Rate (FDR) [@Benjamini1995; @Bates2023] at a pre-defined significance level (e.g., $\alpha \leq 0.1$).<br>
The library integrates with the popular `pyod` library [@Zhao2019; @Zhao2024], making it easy to apply these conformal techniques to a wide range of anomaly detection models.

# Statement of Need

A primary challenge in anomaly detection is setting an appropriate anomaly threshold, which directly impacts the false positive rate.
In high-stakes domains such as fraud detection, medical diagnostics, and industrial quality control, controlling the proportion of false positives is crucial, as frequent false alarms can lead to *alert fatigue* and render a system impractical.
The `nonconform` package addresses this by replacing raw anomaly scores with $p$-values, which enables formal FDR control.
This makes the conformal methods *threshold-free*, as decision thresholds are a direct result of respective statistical procedures.

$$
FDR = \frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
[@Benjamini1995; @Benjamini2009]


Moreover, conformal methods are *non-parametric* and *model-agnostic*, making them compatible with any model that produces consistent anomaly scores.
The `nonconform` package provides a range of strategies for creating the calibration set from training data, even in low-data regimes [@Hennhofer2024].
With the gathered calibration set, the package can compute standard conformal $p$-values or modified *weighted* conformal $p$-values [@Jin2023] for test data.
Weighted $p$-values are particularly useful when the statistical assumption of exchangeability is weakened by covariate shift between calibration and test data.
By providing these tools, `nonconform` enables researchers and practitioners to build anomaly detectors whose outputs are statistically controlled to cap the FDR at a desired nominal level:

The core assumption for the methods in `nonconform` is that the data is exchangeable, meaning the joint probability distribution is invariant to the order of observations.
This makes the methods suitable for many cross-sectional data analysis tasks but not for time-series data where temporal ordering is informative.

# Acknowledgements

This work was conducted in part within the research projects *Biflex Industrie* (grant number 01MV23020A) and *AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*).
