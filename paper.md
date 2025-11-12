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

Time series data from environmental monitoring networks and scientific instruments frequently contain gaps due to equipment failures, maintenance periods, or transmission errors. `svd_imputer` is a Python package that imputes missing values in multivariate time series using Singular Value Decomposition (SVD). The package exploits spatial and temporal correlations across multiple series through robust preprocessing, data augmentation techniques, Monte Carlo uncertainty quantification, and automatic rank estimation, while following scikit-learn conventions for easy workflow integration.

# Statement of need

Multivariate time series from environmental monitoring networks often exhibit strong spatial and temporal correlations. Traditional univariate imputation methods ignore these cross-series relationships and provide no uncertainty estimates [@hastie2009elements]. While sophisticated machine learning approaches exist [@stekhoven2012missforest], they often require extensive hyperparameter tuning and substantial computational resources.

Matrix completion methods based on low-rank approximations offer a middle ground: they exploit correlations between series while remaining computationally efficient [@candes2010matrix; @mazumder2010spectral]. However, existing Python implementations require significant customization or lack uncertainty quantification frameworks.

`svd_imputer` addresses these limitations by providing:

- **Scikit-learn-compatible API** [@pedregosa2011scikit] for seamless workflow integration
- **Automatic rank estimation** via variance thresholds or cross-validation
- **Monte Carlo uncertainty quantification** with multiple masking strategies

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

Monte Carlo validation estimates imputation uncertainty [@efron1979bootstrap]:

1. Mask observed values (10% default)
2. Impute artificially missing values
3. Compute error metrics against true values
4. Repeat to build error distributions

Two masking strategies simulate different failure modes: random selection and temporal blocks.

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

# With uncertainty quantification  
df_imputed, uncertainty = imputer.fit_transform(
    return_uncertainty=True,
    n_repeats=100,
    mask_strategy='block'
)

print(f"RMSE: {uncertainty['rmse']:.3f}")
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

`svd_imputer` combines efficient SVD imputation, automatic rank selection, Monte Carlo uncertainty quantification, and a scikit-learn-compatible API in a single package.

# Acknowledgements

We thank the open-source community for feedback on early versions of this package.



# References
