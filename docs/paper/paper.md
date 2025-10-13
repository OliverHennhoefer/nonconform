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

Quantifying uncertainty is fundamental for AI systems in safety-critical, high-cost-of-error domains, as reliable decision-making depends on it. The Python package `nonconform` offers statistically principled uncertainty quantification for semi-supervised anomaly detection based on one-class classification [@Tax2001]. It implements methods from conformal anomaly detection [@Laxhammar2010; @Bates2023; @Jin2023], grounded in conformal inference [@Papadopoulos2002; @Vovk2005; @Lei2012].

The package `nonconform` calibrates anomaly detection models to produce statistically valid $p$-values from raw anomaly scores. Conformal calibration uses a hold-out set $\mathcal{D}_{\text{calib}}$ of size $n$ containing normal instances, while the model is trained on a separate normal dataset. For a new observation $X_{n+1}$ with anomaly score $\hat{s}(X_{n+1})$, the $p$-value is computed by comparing this score to the empirical distribution of calibration scores $\hat{s}(X_i)$ for $i \in \mathcal{D}_{\text{calib}}$. The conformal $p$-value $\hat{u}(X_{n+1})$ is defined as the normalized rank of $\hat{s}(X_{n+1})$ among the calibration scores [@Liang2024]:

$$
\hat{u}(X_{n+1}) \;=\; \frac{\lvert \{ i \in \mathcal{D}_{\text{calib}} : \hat{s}(X_i) \leq \hat{s}(X_{n+1}) \} \rvert}{n}.
$$

By framing anomaly detection as a sequence of statistical hypothesis tests, these $p$-values enable systematic control of the *marginal* (average) false discovery rate (FDR) [@Benjamini1995; @Bates2023] at a predefined significance level via appropriate statistical procedures.
The library integrates seamlessly with the widely used `pyod` library [@Zhao2019; @Zhao2024], extending conformal techniques to a broad range of anomaly detection models.

# Statement of Need

A major challenge in anomaly detection lies in setting an appropriate anomaly threshold, as it directly influences the false positive rate. In high-stakes domains such as fraud detection, medical diagnostics, and industrial quality control, excessive false alarms can lead to *alert fatigue* and render systems impractical.  
The package `nonconform` mitigates this issue by replacing raw anomaly scores with $p$-values, enabling formal control of the FDR. Consequently, conformal methods become effectively *threshold-free*, since anomaly thresholds are implicitly determined by underlying statistical procedures.

$$
FDR = \frac{\text{Efforts Wasted on False Alarms}}{\text{Total Efforts}}
$$
[@Benjamini2009]

Conformal methods are *nonparametric* and *model-agnostic*, applying to any model that produces consistent anomaly scores on arbitrarily distributed data. Their key requirement is the assumption of *exchangeability* between calibration and test data, ensuring the validity of resulting conformal $p$-values.  
Exchangeability only requires that the joint data distribution is invariant under permutations, making it more general---and less restrictive---than the independent and identically distributed (*i.i.d.*) assumption common in classical machine learning.

To operationalize this assumption, `nonconform` constructs calibration sets from training data using several strategies, including approaches for low-data regimes [@Hennhofer2024] that do not require a dedicated hold-out set. Based on these calibration sets, the package computes *standard* or *weighted* conformal $p$-values [@Jin2023], which are particularly useful under covariate shift, when exchangeability is only approximate.  
These tools enable practitioners to build anomaly detectors whose outputs are statistically controlled to maintain the FDR at a chosen nominal level.

Overall, reliance on exchangeability makes these methods well-suited to cross-sectional data but less appropriate for time series applications, where temporal ordering conveys essential information.


# Acknowledgements

This work was conducted in part under the research projects *Biflex Industrie* (Grant no. 01MV23020A) funded by the *German Federal Ministry of Economic Affairs and Climate Action*.

# References
