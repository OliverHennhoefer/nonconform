---
title: 'nonconform: Conformal Anomaly Detection'
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

The ability to quantify uncertainty represents a fundamental requirement for AI systems operating in safety-critical domains.
In context of anomaly detection, this directly translates to controlling the rate of False Positive (_Type I Error_) while preserving sensitivity to genuine anomalies.
**Conformal Anomaly Detection** [@Laxhammar2010] emerges as a promising approach for providing respective statistical guarantees by calibrating a given detector model in order to align anomaly scores with their empirical false alarm rates.
Instead of relying on anomaly scores and arbitrarily set thresholds, this approach converts the anomaly scores to statistically valid $p$-values that can then be adjusted by statistical methods that control the False Discovery Rate (FDR) [@Benjamini1995] within a set of tested instances [@Bates2023].

The Python library `nonconform` is an open-source software package that provides a range of tools to enable conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012] for one-class classification [@Petsche1994]. The library computes classical and weighted conformal $p$-values [@Jin2023] using different conformalization strategies that make them suitable for application even in low-data regimes [@Hennhofer2024]. The library integrates with the majority of `pyod` anomaly detection models [@Zhao2019; @Zhao2024].

# Statement of Need

The field of anomaly detection comprises methods for identifying observations that either deviate from the majority of observations or otherwise do not *conform* to an expected state of *normality*. The typical procedure leverages anomaly scores and thresholds to distinguish in-distribution data from out-of-distribution data. However, this approach does not provide statistical guarantees regarding its estimates. A major concern in anomaly detection is the rate of False Positives among proclaimed discoveries. Depending on the domain, False Positives can be expensive. Triggering *false alarms* too often results in *alert fatigue* and eventually renders the detection system ineffective and impractical.

In such contexts, it is necessary to control the proportion of False Positives relative to the entirety of proclaimed discoveries (the number of triggered alerts). In practice, this is measured by the FDR, which translates to:
$$
FDR=\frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
[@Benjamini1995; @Benjamini2009].

Framing anomaly detection tasks as sets of statistical hypothesis tests, with $H_0$ claiming that the data is *normal* (no *discovery* to be made), enables controlling the FDR when statistically valid $p$-values (or test statistics) are available. When conducting multiple *simultaneous* hypothesis tests, it is furthermore necessary to *adjust* for multiple testing, as fixed *significance levels* (typically $\alpha \leq 0.05$) would lead to inflated overall error rates.

The `nonconform` (*<ins>non</ins>-<ins>conform</ins>ity-based anomaly detection*) package provides the tools necessary for creating anomaly detectors whose outputs can be statistically controlled to cap the FDR at a nominal level among normal instances under exchangeability. It provides wrappers for a wide range of anomaly detectors (e.g., [Variational-]Autoencoder, IsolationForest, One-Class SVM) complemented by a rich range of conformalization strategies (mostly depending on the *data regime*) to compute classical conformal $p$-values or modified *weighted* conformal $p$-values. The need for *weighted* conformal $p$-values arises when the underlying statistical assumption of *exchangeability* is violated due to covariate shift between calibration and test data. Finally, `nonconform` offers built-in statistical adjustment measures like Benjamini-Hochberg [@Benjamini1995] that correct obtained and statistically valid $p$-values for the multiple testing problem when testing a *batch* of observations simultaneously.

# Acknowledgements

This work was conducted in part within the research projects *Biflex Industrie* (grant number 01MV23020A) and *AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*).
