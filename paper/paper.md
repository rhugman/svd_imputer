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
  - name: Marti Burcet
    orcid: 0000-0002-7422-3368
    affiliation: 2
  - name: Tao Cui
    orcid: 0000-0001-9853-9423
    affiliation: 2
affiliations:
 - name: INTERA, Portugal
   index: 1
 - name: Office of Groundwater Impact Assessment, Brisbane, QLD, Australia
   index: 2   
date: 12 November 2025
bibliography: paper.bib
---

# Summary

Time series data from environmental monitoring networks and scientific instruments frequently contain gaps due to equipment failures, maintenance periods, or transmission errors. `svd_imputer` is a Python package that imputes missing values in multivariate time series using Singular Value Decomposition (SVD). The package implements an Expectation-Maximization (EM) workflow to iteratively fill in missing data with an optimal low-rank approximation. Multiple imputation and bootstrap workflows are implemented to estimate imputation uncertainty. The package provides an easy-to-use and computationally frugal tool for filling time series data gaps.

# Statement of Need

Data gaps in monitored environmental variables can pose challenges for hydrological and hydrogeological studies. For example:
 - Calculation of long-term drought indices often requires regular samples and can be affected by data gaps during extreme periods (e.g., droughts and floods).
 - Interpolating spatial distributions requires data during the same period.
 - Numerical model boundary conditions often rely on spatio-temporally varying inputs interpolated between monitored locations.

Broadly, these challenges stem from the inherent inconsistency of distributed monitoring networks. Raw environmental observations are frequently fragmented, varying in duration, continuity, and measurement quality. Because the accurate representation of physical processes (such as river–aquifer interaction) depends on a seamless spatio-temporal distribution of data, a systematic preprocessing and data harmonization phase is often a prerequisite for analysis.

Where time series gaps affect the outcomes of an analysis, robust approaches to filling the gaps and estimating the uncertainty of the filling method are required. Recently, @burcet2025iah introduced an SVD-based workflow specifically designed for this purpose. 

The underlying principle of this method is to extract the relationships between high-quality measurements across different monitoring stations to fill missing records or replace low-quality data based on those extracted dependencies. This is implemented by iteratively applying Singular Value Decomposition (SVD) to a matrix representing variable magnitudes (such as fluxes or levels) per date.

Multivariate time series from environmental monitoring networks often exhibit strong spatial and temporal correlations, meaning the underlying data matrix is frequently low-rank. Traditional univariate imputation methods ignore these cross-series relationships and provide no uncertainty estimates [@hastie2009elements]. While sophisticated machine learning approaches exist [@stekhoven2012missforest], they often require hyperparameter tuning and more extensive computational resources.

Matrix completion methods based on low-rank approximations offer a middle ground: they exploit correlations between series while remaining computationally efficient [@candes2010matrix; @mazumder2010spectral]. While `scikit-learn` [@pedregosa2011scikit] provides `IterativeImputer`, it lacks a native SVD-based engine optimized for the rank-deficient matrices common in environmental data and does not natively provide robust uncertainty quantification.

The `svd_imputer` package implements the workflow introduced by [@burcet2025iah], extends it to include uncertainty quantification, and addresses limitations in existing Python implementations by providing:

- **Automatic rank estimation** via variance thresholds or cross-validation.
- **Uncertainty quantification** via Multiple Imputation and Rubin's Rules [@rubin1987multiple], allowing practitioners to propagate imputation error into downstream physical models.
- **Time-series specific augmentation utilities**, such as lag and derivative features, to capture temporal dynamics within the SVD framework.

### Comparison with Existing Tools

| Feature | `svd_imputer` | `scikit-learn` | `Amelia` (R) | `metan` (R) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Engine** | SVD / EM | MICE / Iterative | EM / Bootstrapping | EM-SVD |
| **Uncertainty** | Rubin's Rules | No | Yes | No |
| **Augmentation**| Lag & Derivative | No | No | No |
| **Language** | Python | Python | R | R (Archived) |

# Implementation

`svd_imputer` implements iterative SVD imputation for matrix completion [@troyanskaya2001missing; @mazumder2010spectral]. Data is represented as a matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, where $n$ represents time entries and $p$ represents monitored sites.

The algorithm assumes $\mathbf{X}$ can be approximated by a low-rank matrix:
$$\mathbf{X} \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^T$$
where subscript $r$ denotes truncation to rank $r$.

### The Iterative Algorithm
1.  **Initialize**: Fill missing entries $(i,j) \notin \Omega$ (where $\Omega$ is the set of observed indices) with column means to create $\mathbf{X}^{(0)}$.
2.  **Iterate until convergence**:
    - Compute SVD of the current matrix: $\mathbf{X}^{(t-1)} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$.
    - Construct the low-rank reconstruction: $\mathbf{X}_{rec} = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^T$.
    - Update only the missing entries: $X_{ij}^{(t)} = (X_{rec})_{ij}$ for all $(i,j) \notin \Omega$, while keeping $X_{ij}^{(t)} = X_{ij}^{(0)}$ for all $(i,j) \in \Omega$.
3.  **Stop**: When the change in the missing entries stabilizes, defined by the Frobenius norm: $\|\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)}\|_F < \epsilon$.

### Uncertainty Quantification

1.  **Multiple Imputation (Stochastic SVD)**: This method injects Gaussian noise based on residual variance during the iterative process. 
    - At each iteration, the residual variance $\hat{\sigma}^2$ is calculated from observed data points $\Omega$.
    - Missing entries are updated as $X_{new, ij} = (X_{rec})_{ij} + \mathcal{N}(0, \hat{\sigma}^2)$.
    - Following [@rubin1987multiple], a set of $M$ independent completed matrices (where $M$ is the number of imputations) is aggregated to calculate the final point estimate ($\bar{\theta}$) and Total Variance ($T$), which accounts for both within-imputation and between-imputation variance.

2.  **Monte Carlo Bootstrap-Validation**: Estimates global reconstruction error by repeatedly masking a subset of observed values and imputing them [@efron1979bootstrap]. It supports both "Random" and "Block" masking strategies to simulate sensor outages.

3.  **Monte Carlo Bootstrap-Validation**: Estimates global reconstruction error by repeatedly masking a subset of observed values and imputing them [@efron1979bootstrap]. It supports both "Random" and "Block" masking strategies to simulate sensor outages.

# Example Usage

The package is designed using a `scikit-learn` style API, requiring minimal user input. The workflow is handled through the `Imputer` class. The user provides time series data in wide format (i.e., a unique time series per column).

On initialization, data is validated and standardized by default. The user has options on how to determine the SVD rank (manual, variance-based, or through cross-validation). Iterative SVD is undertaken by calling `fit_transform()`. Optionally, uncertainty estimation through multiple imputation can be specified.

```python
import pandas as pd
from svd_imputer import Imputer

# Load time series data
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Basic imputation with automatic rank estimation
imputer = Imputer(data=df, rank='auto')
df_imputed = imputer.fit_transform()

# Multiple Imputation for element-wise uncertainty
df_imputed, df_std = imputer.fit_transform(
    return_uncertainty=True,
    n_imputations=10
)
```

\autoref{fig:1} illustrates the imputer's performance across four synthetic sites with varying correlation structures. The `truth` (solid black) represents the held-back data, while the `imputed` values (dashed red) fill the gaps created for validation solely based on SVD using the `input` data (blue dots). The shaded grey region represents the 95% confidence interval, derived from $M=10$ imputations using Rubin’s Rules to account for both model and residual uncertainty. Note that for Series_D, which lacks correlation with other sites, the gap is successfully recovered using derivative-based and time-lag data augmentation.

![Demonstration of SVD-based imputation. The truth (black) is compared against imputed values (dashed red). The shaded area denotes the 95% confidence interval calculated via Rubin’s Rules, propagating uncertainty from multiple stochastic SVD iterations.\label{fig:1}](fig1.png)


# Final remarks

The `svd_imputer` package offers a streamlined, Python-based workflow designed to bridge the gap between complex matrix completion theory and practical environmental application. By providing a ready-to-use pipeline, it allows practitioners to implement advanced multivariate imputation without the overhead of building custom software architectures.

Beyond simple data filling, the tool’s ability to generate robust error estimates represents a significant step forward for hydrological modeling. In fields where analyses directly inform risk-based decision-making, understanding the reliability of imputed data is just as important as the data itself. `svd_imputer` ensures that this uncertainty is no longer an afterthought but a core component of the data preparation process.

Future improvements include exploring approaches to integrating temporal-covarince in informing imputed values, implementing "round robin" or "sequential imputation" workflows to maximize information gain whilst minizing the inflluence of noise.

# Acknowledgements

We thank the JOSS editors and reviewers for their constructive comments improving both the package and paper. 



# References
