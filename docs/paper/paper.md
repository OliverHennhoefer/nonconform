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

`nonconform` is a Python package that implements conformal anomaly detection methods [@Laxhammar2010] to provide statistically principled uncertainty quantification for unsupervised anomaly detection.
The ability to quantify uncertainty is a fundamental requirement for AI systems in safety-critical domains, where reliable decision-making is essential.
Based on the principles of conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012], `nonconform` converts the anomaly scores from underlying anomaly detection models into statistically valid $p$-values.
These $p$-values can be systematically adjusted using methods that control the False Discovery Rate (FDR) [@Benjamini1995; @Bates2023], allowing users to move beyond arbitrary thresholds.
By calibrating detector models to align anomaly scores with their empirical false alarm rates, this approach provides statistical guarantees.
The library integrates with the popular `pyod` library [@Zhao2019; @Zhao2024], making it easy to apply these techniques to a wide range of anomaly detection models.

# Statement of Need

A primary challenge in anomaly detection is managing the rate of false positives.
In many applications, frequent false alarms can lead to *alert fatigue*, ultimately rendering a detection system impractical.
For research in areas such as fraud detection, medical diagnostics, and industrial quality control, controlling the proportion of false positives is crucial.
`nonconform` addresses this by framing anomaly detection as a set of statistical hypothesis tests, enabling the control of the False Discovery Rate (FDR).

The package provides wrappers for various anomaly detection models (e.g., Autoencoder, IsolationForest, One-Class SVM) and includes a range of conformalization strategies.
These strategies can compute classical conformal $p$-values or modified *weighted* conformal $p$-values [@Jin2023], which are suitable even in low-data regimes [@Hennhofer2024].
The need for weighted conformal $p$-values arises when the statistical assumption of exchangeability is violated due to covariate shift between calibration and test data.
By providing these tools, `nonconform` enables researchers to create anomaly detectors with outputs that can be statistically controlled to cap the FDR at a nominal level.
The methods support batch, batch-streaming, and online deployments, making them adaptable to various research settings.

The core assumption for the methods in `nonconform` is that the data is exchangeable, meaning the joint probability distribution is invariant to the order of observations.
This assumption is suitable for many cross-sectional data analysis tasks but not for time-series data where temporal ordering is informative.

# Acknowledgements

This work was conducted in part within the research projects *Biflex Industrie* (grant number 01MV23020A) and *AutoDiagCM* (grant number 03EE2046B) funded by the *German Federal Ministry of Economic Affairs and Climate Action* (*BMWK*).
