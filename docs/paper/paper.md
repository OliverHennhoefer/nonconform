---
title: 'nonconform: Conformal Anomaly Detection (Python)'
tags:
  - Python
  - anomaly detection
  - conformal inference
  - conformal anomaly detection
  - uncertainty quantification
  - false discovery rate
authors:
  - name: Oliver Hennhöfer
    orcid: 0000-0001-9834-4685
    affiliation: 1
affiliations:
 - name: Intelligent Systems Research Group (ISRG), Karlsruhe University of Applied Sciences (HKA), Karlsruhe, Germany
   index: 1
date: 29 May 2025
bibliography: paper.bib
---

# Summary

# Summary

The ability to quantify uncertainty represents a fundamental requirement for AI systems operating in safety-critical or high-stakes domains and is essential for reliable decision-making.
The software package `nonconform` addresses this challenge in the context of unsupervised anomaly detection in form of one-class classification problems [@Petsche1994].
Specifically, the package implements methods from conformal anomaly detection [@Laxhammar2010],
based on the overarching principles of conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012] for statistically principled uncertainty quantification.

The library integrates with `pyod` [@Zhao2019; @Zhao2024] anomaly detection models and converts anomaly scores to statistically valid $p$-values
that can be systematically adjusted using methods that control the False Discovery Rate (FDR) [@Benjamini1995; @Bates2023].
Rather than relying on anomaly scores and arbitrarily set thresholds, this approach provides statistical guarantees by calibrating detector models to align anomaly scores with their empirical false alarm rates.

# Statement of Need

The field of anomaly detection comprises methods for identifying observations that either deviate from the majority of observations or otherwise do not *conform* to an expected state of *normality*.
The typical procedure leverages anomaly scores and thresholds to distinguish in-distribution data from out-of-distribution data.
However, this approach does not provide statistical guarantees regarding its estimates.
A major concern in anomaly detection is the rate of False Positives among proclaimed discoveries.
Depending on the domain, False Positives can be expensive. Triggering *false alarms* too often results in *alert fatigue* and eventually renders the detection system ineffective and impractical.

In the context of anomaly detection, uncertainty quantification directly translates to controlling the rate of False Positive (*Type I Error*) while preserving sensitivity to genuine anomalies.
In practice, it is necessary to control the proportion of False Positives relative to the entirety of proclaimed discoveries (the number of triggered alerts), measured by the FDR:

$$
FDR=\frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
[@Benjamini1995; @Benjamini2009].

Framing anomaly detection tasks as sets of statistical hypothesis tests, with $H_0$ claiming that the data is *normal* (no *discovery* to be made),
enables controlling the FDR when statistically valid $p$-values (or test statistics) are available.
When conducting multiple *simultaneous* hypothesis tests, it is furthermore necessary to *adjust* for multiple testing,
as fixed *significance levels* (typically $\alpha \leq 0.05$) would lead to inflated overall error rates.

The `nonconform` (*<ins>non</ins>-<ins>conform</ins>ity-based anomaly detection*) package provides the tools necessary for creating anomaly detectors
whose outputs can be statistically controlled to cap the FDR at a nominal level among normal instances under exchangeability.
It provides wrappers for a wide range of anomaly detectors (e.g., \[Variational-\]Autoencoder, IsolationForest, One-Class SVM)
complemented by a rich range of conformalization strategies to compute classical conformal $p$-values or modified *weighted* conformal $p$-values [@Jin2023]
using different strategies that make them suitable for application even in low-data regimes [@Hennhofer2024].
The need for *weighted* conformal $p$-values arises when the underlying statistical assumption of *exchangeability* is violated due to covariate shift between calibration and test data.

# Areas of Application

The methods implemented in `nonconform` require data to satisfy the statistical assumption of exchangeability—meaning the joint probability distribution remains unchanged under any permutation of the observation order.
Simply put, data points can be shuffled without affecting their statistical properties.
This assumption naturally holds for independent and identically distributed (i.i.d.) data, making the package suitable for cross-sectional data analysis, quality control,
fraud detection, and medical diagnostics where samples are independently collected.
Time-series and autocorrelated data are unsuitable as temporal ordering carries information that would be lost under permutation, violating exchangeability.
However, when exchangeability holds, the methods support both online and batch-streaming deployments through integration with established FDR control methods.
The `onlineFDR` package[^1] enables real-time anomaly detection while maintaining statistical guarantees.

[^1]: https://github.com/OliverHennhoefer/online-fdr

# Acknowledgements

This work was conducted in part within the research projects *Biflex Industrie* (grant number 01MV23020A) and *AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*).
