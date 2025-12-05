---
title: 'svd_imputer: A Python Package for Time Series Imputation Using Singular Value Decomposition'
tags:
  - Python
  - time series
  - missing data
  - imputation
  - SVD
  - matrix completion
  - uncertainty quantification
authors:
  - name: Rui Hugman
    orcid: 0000-0003-0891-3886
    affiliation: 1
affiliations:
 - name: INTERA, Portugal
   index: 1
date: 12 November 2025
bibliography: paper.bib
---

# Summary

Time series data from environmental monitoring networks and scientific instruments frequently contain gaps due to equipment failures, maintenance periods, or transmission errors. `svd_imputer` is a Python package that imputes missing values in multivariate time series using Singular Value Decomposition (SVD). The package exploits spatial and temporal correlations across multiple series through robust preprocessing, data augmentation techniques, uncertainty quantification via Multiple Imputation, and automatic rank estimation, while following scikit-learn conventions for easy workflow integration.

# Statement of need

Multivariate time series from environmental monitoring networks often exhibit strong spatial and temporal correlations. Traditional univariate imputation methods ignore these cross-series relationships and provide no uncertainty estimates [@hastie2009elements]. While sophisticated machine learning approaches exist [@stekhoven2012missforest], they often require extensive hyperparameter tuning and substantial computational resources.

Matrix completion methods based on low-rank approximations offer a middle ground: they exploit correlations between series while remaining computationally efficient [@candes2010matrix; @mazumder2010spectral]. However, existing Python implementations require significant customization or lack uncertainty quantification frameworks.

`svd_imputer` addresses these limitations by providing:

- **Scikit-learn-compatible API** [@pedregosa2011scikit] for seamless workflow integration
- **Automatic rank estimation** via variance thresholds or cross-validation
- **Uncertainty quantification** via Multiple Imputation and Rubin's Rules

The package targets practitioners working with environmental monitoring data where uncertainty quantification is essential. While similar functionality exists in R packages [@amelia; @buuren2011mice], Python implementations with comparable features have been lacking.

# Implementation

`svd_imputer` implements iterative SVD imputation for matrix completion [@troyanskaya2001missing; @mazumder2010spectral]. The algorithm assumes a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ can be approximated by a low-rank matrix:

$$\mathbf{X} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$$

where subscript $r$ denotes truncation to rank $r$.

The iterative algorithm:

1. Initialize missing values with column means
2. Iterate until convergence:
   - Compute SVD: $\mathbf{X}^{(t-1)} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
   - Update missing entries with rank-$r$ approximation
3. Stop when $\|\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)}\|_F < \epsilon$

## Preprocessing and Data Validation

The package includes comprehensive data preprocessing capabilities:

- **Validation**: Ensures datetime index, sorted data, and sufficient observations
- **Standardization**: Column-wise standardization with NaN-aware calculations  
- **Trend handling**: Optional detrending functionality for non-stationary series

## Data Augmentation

A key feature is data augmentation to improve imputation quality by incorporating temporal structure:

- **Derivative augmentation**: Adds first and second differences to capture rate-of-change patterns
- **Symmetric lag augmentation**: Includes past and future values for interpolation scenarios
- **Asymmetric lag augmentation**: Uses only historical lags for forecasting applications

These augmentation strategies expand the feature space to better capture temporal dependencies in the SVD decomposition.

## Rank Selection

Three methods for rank selection:

- **Variance threshold** (default): Minimum rank explaining 95% of variance
- **Cross-validation**: Optimize rank by minimizing prediction error  
- **Fixed rank**: User-specified value

## Uncertainty Quantification

The package provides two complementary approaches for uncertainty quantification:

1.  **Multiple Imputation (Stochastic SVD)**: Generates element-wise uncertainty intervals for missing values using Rubin's Rules [@rubin1987multiple]. This method injects Gaussian noise based on residual variance during the iterative process to create multiple independent imputations, capturing both model and missing data uncertainty.

    **The Stochastic Algorithm**:
    The standard SVD update loop is modified to include a stochastic step.
    
    - **Residual Estimation**: At each iteration, calculate the variance of the error on the observed data points $\Omega$:
      $$\hat{\sigma}^2 = \frac{1}{| \Omega |} \sum_{(i,j) \in \Omega} (X_{obs, ij} - (U \Sigma V^T)_{ij})^2$$
    
    - **Stochastic Imputation**: Fill missing entries $(i, j)$ with the prediction plus random noise drawn from the residual distribution:
      $$X_{new, ij} = (U \Sigma V^T)_{ij} + \mathcal{N}(0, \hat{\sigma}^2)$$
    
    This process is repeated to generate $M$ independent completed matrices.

    **Aggregation (Rubin's Rules)**:
    The $M$ completed matrices are aggregated to obtain the final estimate and uncertainty bound:
    
    - **Final Point Estimate** ($\bar{\theta}$): The average of the $M$ imputed values.
      $$\bar{\theta} = \frac{1}{M} \sum_{m=1}^M \hat{\theta}_m$$
    
    - **Total Variance** ($T$): Combines the Within-Imputation Variance ($W$, average noise level) and Between-Imputation Variance ($B$, variance across datasets):
      $$T = W + \left(1 + \frac{1}{M}\right)B$$
      where $W = \frac{1}{M} \sum_{m=1}^M \hat{\sigma}^2_m$ and $B = \frac{1}{M-1} \sum_{m=1}^M (\hat{\theta}_m - \bar{\theta})^2$.

2.  **Monte Carlo Validation**: Estimates global reconstruction error (RMSE, MAE) by repeatedly masking a subset of observed values and imputing them [@efron1979bootstrap]. This approach is primarily used for model validation and automatic rank selection via cross-validation.

    **The Validation Algorithm**:
    For each repetition $k=1 \dots K$:
    
    - **Masking**: Generate a test set $\Omega^{(k)}_{test}$ by hiding a fraction $f$ (e.g., 10%) of the originally observed values $\Omega$.
      - *Random Strategy*: Indices are selected uniformly at random.
      - *Block Strategy*: Contiguous temporal blocks are selected to simulate sensor outages.
      
    - **Imputation**: Apply the deterministic SVD imputer to the masked dataset to obtain the reconstruction $\hat{X}^{(k)}$.
    
    - **Error Calculation**: Compute the discrepancy between the imputed values and the ground truth for the masked entries:
      $$RMSE_k = \sqrt{\frac{1}{|\Omega^{(k)}_{test}|} \sum_{(i,j) \in \Omega^{(k)}_{test}} (X_{ij} - \hat{X}^{(k)}_{ij})^2}$$

    **Aggregation**:
    The final performance metric is the average error over all $K$ repetitions, providing a robust estimate of the model's generalization capability:
    $$RMSE_{total} = \frac{1}{K} \sum_{k=1}^K RMSE_k$$

# Example Usage

Basic usage with automatic rank estimation:

```python
import pandas as pd
from svd_imputer import Imputer
from svd_imputer.preprocessing import (
    create_derivative_augmented_matrix,
    create_symmetric_augmented_matrix
)

# Load time series data
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Basic imputation
imputer = Imputer(data=df, variance_threshold=0.95)
df_imputed = imputer.fit_transform()

# Data augmentation for better temporal structure capture
df_aug = create_derivative_augmented_matrix(df)
imputer_aug = Imputer(data=df_aug, variance_threshold=0.95)
df_aug_imputed = imputer_aug.fit_transform()

# 1. Multiple Imputation (for element-wise uncertainty)
df_imputed, df_uncertainty = imputer.fit_transform(
    return_uncertainty=True,
    n_imputations=10
)
print(f"Average uncertainty: {df_uncertainty.mean().mean():.3f}")

# 2. Monte Carlo Validation (for global error estimation)
imputer.fit()
validation_results = imputer.estimate_uncertainty(
    n_repeats=100, 
    mask_strategy='block'
)
print(f"Estimated RMSE: {validation_results['RMSE']['mean']:.3f}")
```

The augmentation functions enable users to leverage temporal patterns for improved imputation accuracy in time series with strong sequential dependencies.

# Implementation details

`svd_imputer` uses NumPy [@harris2020array] for computations, pandas [@reback2020pandas] for time series handling, and follows scikit-learn [@pedregosa2011scikit] conventions. The package features:

- **Modular design**: Separate modules for imputation and preprocessing with robust data validation
- **Comprehensive testing**: 86 unit and integration tests covering edge cases
- **Data-centric API**: Validation and preprocessing performed once at initialization
- **Flexible augmentation**: Three augmentation strategies for different temporal scenarios

The preprocessing module handles common time series challenges including irregular sampling, missing validation, and standardization with NaN-aware statistics. The augmentation functions create expanded feature matrices that better capture temporal dependencies, particularly useful for datasets with strong sequential patterns or when interpolating gaps within series rather than extrapolating at boundaries.

The core SVD computation uses NumPy's optimized LAPACK routines. Computational complexity is $O(k \cdot n \cdot p \cdot \min(n,p))$ where $k$ is iterations (typically 10-50). The package handles datasets up to ~10,000 time points and ~100 variables efficiently on standard hardware.

# Comparison with existing tools

Existing imputation packages have limitations:

- **scikit-learn** [@pedregosa2011scikit]: `SimpleImputer` and `IterativeImputer` lack uncertainty quantification
- **fancyimpute**: No longer maintained
- **Amelia** [@amelia] (R): Uncertainty estimation but limited model assumptions  
- **mice** [@buuren2011mice] (R): Comprehensive but computationally intensive

`svd_imputer` combines efficient SVD imputation, automatic rank selection, Multiple Imputation for uncertainty quantification, and a scikit-learn-compatible API in a single package.

# Acknowledgements

We thank the open-source community for feedback on early versions of this package.



# References
