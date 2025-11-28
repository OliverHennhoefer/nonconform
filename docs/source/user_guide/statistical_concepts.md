# Statistical Concepts in nonconform

This guide explains the key statistical concepts that underpin nonconform's functionality.

## Conformal Inference

Conformal inference is a framework for uncertainty quantification that provides valid prediction intervals and p-values without making strong distributional assumptions [[Vovk et al., 2005](#references); [Angelopoulos & Bates, 2023](#references)]. In the context of anomaly detection [[Bates et al., 2023](#references)], it allows us to:

1. Convert raw anomaly scores into statistically valid p-values
2. Control false discovery rates at specified levels
3. Provide uncertainty quantification for anomaly detection

### Exchangeability

The key assumption in conformal inference is exchangeability [[Vovk et al., 2005](#references)], which is weaker than independence. Data points are exchangeable if their joint distribution is invariant to permutations. This means:

- The order of the data points doesn't matter
- Each data point is treated equally in the analysis
- The statistical guarantees hold under this assumption

### P-values in Anomaly Detection

In nonconform, p-values represent the probability of observing a more extreme anomaly score under the null hypothesis (that the point is normal). Specifically:

- Small p-values indicate strong evidence against the null hypothesis
- Large p-values suggest the point is likely normal
- The p-values are valid in the sense that under the null hypothesis, they are stochastically larger than uniform

## False Discovery Rate (FDR) Control

FDR control is a multiple testing procedure that limits the expected proportion of false discoveries among all discoveries [[Benjamini & Hochberg, 1995](#references)]. nonconform implements the Benjamini-Hochberg procedure, which:

1. Controls FDR at a specified level α
2. Is more powerful than family-wise error rate control
3. Provides valid inference even when tests are dependent

### How FDR Control Works

1. Sort p-values in ascending order
2. Find the largest p-value that satisfies p(i) ≤ (i/m)α
3. Reject all null hypotheses with p-values less than or equal to this threshold

## Weighted Conformal p-values

When the exchangeability assumption is violated (e.g., due to covariate shift), weighted conformal p-values can be used [[Jin & Candès, 2023](#references); [Tibshirani et al., 2019](#references)]. These:

1. Account for differences between training and test distributions
2. Maintain statistical validity under weaker assumptions
3. Can improve power in the presence of distributional shifts

## Calibration and Validation

The calibration process in nonconform involves:

1. Splitting the data into training and calibration sets
2. Computing nonconformity scores on the calibration set
3. Using these scores to calibrate the p-values for new observations

This process ensures that the resulting p-values are valid and can be used for statistical inference.

## Statistical Guarantees

nonconform provides the following statistical guarantees:

1. **Marginal Validity**: P-values are valid marginally over the calibration set
2. **FDR Control**: The Benjamini-Hochberg procedure controls FDR at the specified level
3. **Power**: The methods are designed to maximize power while maintaining validity

## Best Practices

For optimal statistical performance:

1. Use sufficient calibration data (typically >1000 points)
2. Ensure the calibration data is representative of the normal class
3. Consider using resampling strategies in low-data regimes
4. Use weighted conformal p-values when dealing with distributional shifts
5. Validate the exchangeability assumption when possible

## References

- **Vovk, V., Gammerman, A., & Shafer, G. (2005)**. *Algorithmic Learning in a Random World*. Springer. [Foundational theory of conformal prediction and exchangeability]

- **Bates, S., Candès, E., Lei, L., Romano, Y., & Sesia, M. (2023)**. *Testing for Outliers with Conformal p-values*. The Annals of Statistics, 51(1), 149-178. [Application of conformal prediction to anomaly detection]

- **Angelopoulos, A. N., & Bates, S. (2023)**. *Conformal Prediction: A Gentle Introduction*. Foundations and Trends in Machine Learning, 16(4), 494-591. [Comprehensive modern introduction to conformal prediction]

- **Benjamini, Y., & Hochberg, Y. (1995)**. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B, 57(1), 289-300. [FDR control methodology]

- **Jin, Y., & Candès, E. J. (2023)**. *Model-free Selective Inference Under Covariate Shift via Weighted Conformal p-values*. Biometrika, 110(4), 1090-1106. arXiv:2307.09291. [Weighted conformal inference under covariate shift]

- **Tibshirani, R. J., Barber, R. F., Candes, E., & Ramdas, A. (2019)**. *Conformal Prediction Under Covariate Shift*. Advances in Neural Information Processing Systems, 32. arXiv:1904.06019. [Early work on conformal prediction with distribution shift]

## Next Steps

- See [conformal inference](conformal_inference.md) for detailed theoretical foundations
- Learn about [weighted conformal p-values](weighted_conformal.md) for handling distribution shift
- Explore [conformalization strategies](conformalization_strategies.md) for different data scenarios
- Check [input validation](input_validation.md) for parameter constraints 