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

Time series data from environmental monitoring networks and scientific instruments frequently contain gaps due to equipment failures, maintenance periods, or transmission errors. `svd_imputer` is a Python package that imputes missing values in multivariate time series using Singular Value Decomposition (SVD). The package implements an Expectation-Maximization (EM) workflow to iteratively fill in missing data with an optimal low-rank approximation. Multiple imputation and bootstrap workflows are implemented to estimate imputation uncertainty. The package provides an easy to use and computationaly frugal tool for filling time series data gaps.

# Statement of need

Data gaps in monitored envirnomental variables can pose chalenges for hydrological and hydrogeologicl studies. For example:
 - Calculation of long-term drought indices often require regular samples, and can be affected by data gaps during extreme periods (e.g., droughts and floods).
 - Interpolating spatial distributions require data during the same period.
 - Numerical model boundary conditions often rely on spatio-temporal varying inputs, interpolated between monitorined locations.

Broadly, these challenges stem from the inherent inconsistency of distributed monitoring networks. Raw environmental observations are frequently fragmented, varying in duration, continuity, and measurement quality. Because the accurate representation of physical processes (such as river–aquifer interaction) depends on a seamless spatio-temporal distribution of data, a systematic preprocessing and data harmonization phase is often a prerequisite for analysis.

Where time series gaps affect the outcomes of an analysis, robust approaches to filling the gaps and estimating the uncertainty of the filling method are required. Recently, [@burcet2025iah] introduced an SVD-based workflow specifically designed for this purpose.

The underlying principle of this method is to extract the relationships between high-quality measurements across different monitoring stations to fill missing records or replace low-quality data based on those extracted dependencies. This is implemented by iteratively applying Singular Value Decomposition (SVD) to a matrix representing variable magnitudes (such as fluxes or levels) per date.

From the SVD factorization, a low-rank approximation of the data for all stations and times is calculated. This approximation effectively filters out localized noise and preserves only the dominant spatio-temporal trends that best represent the monitoring network as a whole. Finally, this low-rank structure is used to impute missing values and replace erroneous records, ensuring a physically consistent dataset.

Multivariate time series from environmental monitoring networks often exhibit strong spatial and temporal correlations, meaning the underlying data matrix is frequently low-rank. Traditional univariate imputation methods ignore these cross-series relationships and provide no uncertainty estimates [@hastie2009elements]. While sophisticated machine learning approaches exist [@stekhoven2012missforest], they often require extensive hyperparameter tuning and substantial computational resources.

Matrix completion methods based on low-rank approximations offer a middle ground: they exploit correlations between series while remaining computationally efficient [@candes2010matrix; @mazumder2010spectral]. While `scikit-learn` [@pedregosa2011scikit] provides `IterativeImputer`, it lacks a native SVD-based engine optimized for the rank-deficient matrices common in environmental data and does not natively provide robust uncertainty quantification.


The `svd_imputer` package implements the workflow introduced by [@burcet2025iah], extends it to include uncertainty quantification abd addresses limitations in existing Python implementations by providing:

- **Automatic rank estimation** via variance thresholds or cross-validation.
- **Uncertainty quantification** via Multiple Imputation and Rubin's Rules [@rubin1987multiple], allowing practitioners to propagate imputation error into downstream physical models.
- **Time-series specific augmentation utilities**, such as lag and derivative features, to capture temporal dynamics within the SVD framework.

# Implementation

`svd_imputer` implements iterative SVD imputation for matrix completion [@troyanskaya2001missing; @mazumder2010spectral]. Data is represented as a matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, where $n$ represents time entries and $p$ represents monitored sites.

The algorithm assumes $\mathbf{X}$ can be approximated by a low-rank matrix:
$$\mathbf{X} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$$
where subscript $r$ denotes truncation to rank $r$.

### The Iterative Algorithm
1.  **Initialize**: Fill missing entries $(i,j) \notin \Omega$ (where $\Omega$ is the set of observed indices) with column means.
2.  **Iterate until convergence**:
    - Compute SVD of the current matrix: $\mathbf{X}^{(t-1)} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.
    - Construct the reconstruction: $\mathbf{X}_{rec} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$.
    - Update missing entries: $X_{ij}^{(t)} = (X_{rec})_{ij}$ for all $(i,j) \notin \Omega$.
3.  **Stop**: When $\|\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)}\|_F < \epsilon$.

### Uncertainty Quantification

1.  **Multiple Imputation (Stochastic SVD)**: This method injects Gaussian noise based on residual variance during the iterative process. 
    - At each iteration, the residual variance $\hat{\sigma}^2$ is calculated from observed data points $\Omega$.
    - Missing entries are updated as $X_{new, ij} = (X_{rec})_{ij} + \mathcal{N}(0, \hat{\sigma}^2)$.
    - Following [@rubin1987multiple], $M$ independent completed matrices are aggregated to calculate the final point estimate ($\bar{\theta}$) and Total Variance ($T$), which accounts for both within-imputation and between-imputation variance.

2.  **Monte Carlo Bootstrap-Validation**: Estimates global reconstruction error by repeatedly masking a subset of observed values and imputing them [@efron1979bootstrap]. It supports both "Random" and "Block" masking strategies to simulate sensor outages.



# Comparison with existing tools

| Feature | `svd_imputer` | `scikit-learn` | `Amelia` (R) | `metan` (R) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Engine** | SVD / EM | MICE / Iterative [@pedregosa2011scikit] | EM / Bootstrapping [@amelia] | EM-SVD [@Olivoto2020] |
| **Uncertainty** | Rubin's Rules | No | Yes | No |
| **Augmentation**| Lag & Derivative | No | No | No |
| **Language** | Python | Python | R | R (Archived) |

# Example Usage

```python
import pandas as pd
from svd_imputer import Imputer

# Load time series data
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Basic imputation with 95% variance threshold
imputer = Imputer(data=df, variance_threshold=0.95)
df_imputed = imputer.fit_transform()

# Multiple Imputation for element-wise uncertainty
df_imputed, df_std = imputer.fit_transform(
    return_uncertainty=True,
    n_imputations=10
)

# Monte Carlo Validation for RMSE estimation
validation = imputer.estimate_uncertainty(n_repeats=50, mask_strategy='block')
print(f"Estimated RMSE: {validation['RMSE']['mean']:.3f}")
```


# Acknowledgements

We thank the open-source community for feedback on early versions of this package.



# References
