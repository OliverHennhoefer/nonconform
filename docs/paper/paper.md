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
 - name: Intelligent Systems Research Group, Karlsruhe University of Applied Sciences, Karlsruhe, Germany
   index: 1
date: 9 October 2025
bibliography: paper.bib
---

# Summary

The ability to quantify uncertainty is a fundamental requirement for AI systems in safety-critical and costly-to-error domains, as reliable decision-making strongly depends on it.
The Python package ``nonconform`` offers statistically principled uncertainty quantification for semi-supervised anomaly detection based on one-class classification [@Petsche1994].
The package implements methods from conformal anomaly detection [@Laxhammar2010; @Bates2023; @Jin2023], grounded in the principles of conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012].

The `nonconform` package calibrates anomaly detection models to produce statistically valid $p$-values from raw anomaly scores.
 The calibration process uses a hold-out set $\mathcal{D}_{\text{calib}}$ of size $n$ containing normal instances, where the model has been trained on a separate set of normal data.
For a new observation $X_{n+1}$ with anomaly score $\hat{s}(X_{n+1})$, this is achieved by comparing it to the empirical distribution of calibration scores $\hat{s}(X_i)$ for $i \in \mathcal{D}_{\text{calib}}$.
The conformal $p$-value $\hat{u}(X_{n+1})$ is then defined as the normalized rank of $\hat{s}(X_{n+1})$ among the calibration scores [@Liang2024]:

$$
\hat{u}(X_{n+1}) \;=\; \frac{\lvert \{ i \in \mathcal{D}_{\text{calib}} : \hat{s}(X_i) \leq \hat{s}(X_{n+1}) \} \rvert}{n}.
$$

By framing anomaly detection as a sequence of statistical hypothesis tests, these $p$-values enable systematic control of the False Discovery Rate (FDR) [@Benjamini1995; @Bates2023] at a pre-defined significance level by respective statistical procedures. <br>
The library integrates seamlessly with the widely used ``pyod`` library [@Zhao2019; @Zhao2024], facilitating the application of conformal techniques across a broad range of anomaly detection models.

# Statement of Need

A central challenge in anomaly detection lies in setting an appropriate detection threshold, as it directly determines the false positive rate.
In high-stakes domains such as fraud detection, medical diagnostics, and industrial quality control, controlling false positives is critical: excessive false alarms can cause *alert fatigue* and ultimately render a system impractical.
The ``nonconform`` package addresses this issue by replacing raw anomaly scores with $p$-values, thereby enabling formal FDR control.
As a result, the conformal methods become effectively *threshold-free*, since decision thresholds are determined by the underlying statistical procedures.

$$
FDR = \frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
[@Benjamini2009]


Moreover, conformal methods are *non-parametric* and *model-agnostic*, and thus apply to any model that produces consistent anomaly scores on arbitrarily distributed data.
A key requirement of these methods is the statistical assumption of exchangeability between calibration and test data, which ensures the validity of conformal $p$-values.
Exchangeability requires only that the joint distribution of data is invariant under permutations, making it considerably more general (and thus less restrictive) than the independent and identically distributed (*i.i.d.*) assumption commonly imposed in classical machine learning.

To operationalize this assumption in practice, the ``nonconform`` package provides several strategies for constructing calibration sets from training data, including approaches tailored to low-data regimes [@Hennhofer2024], that do not rely on a dedicated hold-out set.
Based on obtained calibration sets, the package derives either standard conformal $p$-values or weighted conformal $p$-values [@Jin2023], which are particularly useful under covariate shift when exchangeability is only approximate.
By offering these tools, ``nonconform`` enables researchers and practitioners to build anomaly detectors whose outputs are statistically controlled to maintain the FDR at a chosen nominal level.

Overall, the reliance on exchangeability makes these methods well suited for cross-sectional data, but less appropriate for time-series applications where temporal ordering carries essential information.

# Acknowledgements

This work was conducted in part within the research projects *Biflex Industrie* (grant number 01MV23020A) funded by the *German Federal Ministry of Economic Affairs and Climate Action*.

# References
