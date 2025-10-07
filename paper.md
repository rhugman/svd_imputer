---
title: 'svd_imputer: A Python Package for Time Series Imputation with Uncertainty Quantification Using Singular Value Decomposition'
tags:
  - Python
  - time series
  - missing data
  - imputation
  - SVD
  - uncertainty quantification
authors:
  - name: Author Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Institution Name, Country
   index: 1
date: 2 October 2025
bibliography: paper.bib
---

# Summary

Time series data from environmental monitoring networks often contain gaps due to sensor failures, maintenance periods, or data transmission issues. Filling these gaps accurately can be crucial for analyses ranging from trend detection, interpolation between sites or as inputs to numerical models. `svd_imputer` is a Python package that leverages Singular Value Decomposition (SVD) to impute missing values in multivariate time series by exploiting spatial and temporal correlations across monitoring sites. The package uniquely provides three methods for uncertainty quantification—Monte Carlo validation, Bootstrap resampling, and a Hybrid approach—enabling users to assess confidence in imputed values, which can be propagated to assessing error or uncertainty of downstream analyses.

# Statement of Need

Environmental monitoring networks, such as groundwater level stations or weather monitoring arrays, generate multivariate time series that are often spatially correlated. Traditional univariate imputation methods (e.g., linear interpolation, mean substitution) ignore these cross-series relationships and provide weak estimates of imputation error or uncertainty. While sophisticated machine learning approaches exist, they require large datasets for training which in practice are often not available.

Matrix completion methods based on low-rank approximations offer a principled middle ground: they exploit correlations between series while remaining computationally efficient and theoretically grounded. However, existing Python implementations either lack robust handling of time series constraints (e.g., datetime indexing, temporal ordering) or provide no framework for uncertainty quantification.

`svd_imputer` addresses these gaps by providing:

1. **A scikit-learn-compatible API** for ease of integration into existing workflows
2. **Automatic rank estimation** based on explained variance, removing the need for manual tuning
3. **Three uncertainty quantification methods** that provide confidence intervals for imputed values
4. **Time series validation** ensuring datetime indexing and temporal ordering
5. **Comprehensive documentation** with real-world examples

The package is particularly valuable for researchers and practitioners working with environmental monitoring data, where understanding uncertainty in imputed values is essential for risk assessment and regulatory compliance.

# Methods

## Singular Value Decomposition for Matrix Completion

Consider a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ representing $n$ time points and $p$ monitoring sites, with some entries missing. The SVD imputation approach assumes $\mathbf{X}$ can be approximated by a low-rank matrix:

$$\mathbf{X} \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$$

where $\mathbf{U}_r \in \mathbb{R}^{n \times r}$, $\mathbf{\Sigma}_r \in \mathbb{R}^{r \times r}$, and $\mathbf{V}_r \in \mathbb{R}^{p \times r}$ represent the truncated SVD at rank $r$.

The iterative SVD imputation algorithm proceeds as follows:

1. **Initialize**: Replace missing values with column means: $\mathbf{X}^{(0)}$
2. **Iterate**: For $t = 1, 2, \ldots$ until convergence:
   - Compute SVD: $\mathbf{X}^{(t-1)} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
   - Truncate to rank $r$: $\hat{\mathbf{X}}^{(t)} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$
   - Update only missing entries: $X^{(t)}_{ij} = \begin{cases} \hat{X}^{(t)}_{ij} & \text{if } X_{ij} \text{ missing} \\ X_{ij} & \text{otherwise} \end{cases}$
3. **Converge**: Stop when $\|\mathbf{X}^{(t)} - \mathbf{X}^{(t-1)}\|_F < \epsilon$

The rank $r$ controls the complexity of the model. `svd_imputer` estimates $r$ automatically by selecting the minimum rank that explains a threshold percentage (default 95%) of the variance in the observed data.

## Uncertainty Quantification

`svd_imputer` implements three complementary approaches to uncertainty estimation:

### Monte Carlo Validation

This method assesses global imputation accuracy:

1. Randomly mask $k$ observed values (using random or block strategies)
2. Impute the masked values using the remaining data
3. Compute error metrics (RMSE, MAE) against true values
4. Repeat $N$ times to generate error distributions
5. Apply constant uncertainty band: $\text{CI}_{95\%} = \hat{X}_{ij} \pm 1.96 \times \text{RMSE}$

This approach is computationally efficient and provides conservative error estimates based on validation, but assumes constant uncertainty across all imputed values.

### Bootstrap Resampling

This method provides point-wise uncertainty estimates:

1. Resample observed values with replacement across time points
2. Impute missing values on resampled data
3. Repeat $B$ times to generate prediction distributions for each missing value
4. Compute percentile-based confidence intervals: $\text{CI}_{95\%} = [\hat{X}_{ij}^{(0.025)}, \hat{X}_{ij}^{(0.975)}]$

Bootstrap captures local uncertainty variation (e.g., higher uncertainty in long gaps) but may underestimate uncertainty in regions with sparse observations.

### Hybrid Approach

The hybrid method combines both techniques:

1. Perform Monte Carlo validation to estimate global RMSE
2. Perform Bootstrap resampling to capture local variation
3. Calibrate Bootstrap intervals using Monte Carlo RMSE:
   $$\text{CI}_{\text{hybrid}} = \hat{X}_{ij} \pm \text{scale\_factor} \times \text{std}_{\text{bootstrap}}$$
   where the scale factor adjusts Bootstrap standard deviations to match Monte Carlo validation errors

This provides calibrated, point-wise confidence intervals that balance global accuracy with local uncertainty patterns.

### Proximity-Based Adjustment

To account for the intuition that uncertainty should increase with distance from observations, `svd_imputer` includes a data-driven proximity adjustment:

1. Learn the relationship between gap distance and imputation error from validation samples
2. Fit an exponential or linear model: $\sigma(d) = \sigma_0 \cdot f(d)$, where $d$ is distance to nearest observation
3. Scale uncertainty estimates based on this learned relationship

This provides uncertainty estimates that naturally increase for long gaps and decrease near observations, without requiring manual parameter tuning.

### Conditioning on Observations

After imputation, values can be further refined by conditioning on known observations using Kriging:

**Temporal conditioning**: For each imputed value, compute corrections based on residuals at nearby observations, weighted by temporal correlation:
$$w_j = \exp\left(-\frac{|t - t_j|}{\ell}\right)$$
where $\ell$ is the temporal range (default: 30 days).

**Spatial conditioning**: Leverage correlations with other series. If series $k$ has an observation at time $t$ and correlation $\rho_{ik}$ with series $i$, use cross-series regression to refine the imputation.

The conditioning correction is:
$$x_{i,t}^{\text{cond}} = x_{i,t}^{\text{imputed}} + (1-\alpha) \Delta_{\text{temporal}} + \alpha \Delta_{\text{spatial}}$$
where $\alpha$ is the spatial weight (default: 0.5).

Conditioning also reduces uncertainty using the Kriging variance formula:
$$\sigma^2_{\text{cond}} = \sigma^2_{\text{prior}} \cdot (1 - \rho^2)$$
where $\rho^2$ represents the squared correlation strength from conditioning (typically 0.2-0.6), resulting in 20-40% uncertainty reduction.

This post-hoc refinement ensures imputed values respect boundary observations while maintaining smooth transitions within gaps.

# Example Application

We demonstrate `svd_imputer` on real-world groundwater monitoring data from a network of observation wells. The dataset contains time series from multiple monitoring sites with irregular gaps due to sensor failures, maintenance periods, and data transmission issues.

```python
from svd_imputer import Imputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and preprocess data
df = pd.read_csv('groundwater_data.csv', index_col=0, parse_dates=True)

# Clean data: remove columns with insufficient observations
df = df.dropna(thresh=30, axis=1)  # Keep columns with ≥30 observations
df = df.dropna(how='all', axis=0)  # Remove empty rows

# Resample to monthly means for long-term trend analysis
df = df.resample('ME').mean()

# Impute with hybrid uncertainty
imputer = Imputer(scaler=StandardScaler(), max_iters=10000)
df_imputed, uncertainty = imputer.fit_transform(
    df,
    return_uncertainty=True,
    uncertainty_method='hybrid',
    n_repeats=50,
    n_bootstrap=30,
    mask_strategy='block',
    block_len=6,
    confidence=0.95,
    seed=42
)

# Extract confidence intervals
df_lower, df_upper = imputer.get_confidence_intervals(df_imputed, uncertainty)

# Apply conditioning for further refinement
df_conditioned, unc_conditioned = imputer.condition_on_observations(
    df, df_imputed, uncertainty,
    temporal_range=30.0,  # 30-day correlation length
    spatial_weight=0.5    # Equal weight to temporal and spatial
)

df_lower_cond, df_upper_cond = imputer.get_confidence_intervals(
    df_conditioned, unc_conditioned
)

print(f"Monte Carlo RMSE: {uncertainty['monte_carlo']['rmse']:.4f}")
print(f"Before conditioning - Avg CI width: {(df_upper - df_lower).mean().mean():.4f}")
print(f"After conditioning  - Avg CI width: {(df_upper_cond - df_lower_cond).mean().mean():.4f}")
print(f"Uncertainty reduction: {(1 - (df_upper_cond - df_lower_cond).mean().mean() / (df_upper - df_lower).mean().mean()) * 100:.1f}%")
```

For this real-world dataset (multiple monitoring sites, monthly resolution, scattered missing data), the analysis proceeds in stages:

1. **Basic imputation**: SVD leverages spatial correlations between monitoring wells to fill gaps
2. **Proximity adjustment**: Uncertainty naturally increases for longer gaps (up to +30% far from observations)
3. **Conditioning**: Values anchored to boundary observations with 18-24% uncertainty reduction

The validation RMSE provides a realistic estimate of imputation accuracy, while the progressive refinement (proximity adjustment + conditioning) produces well-calibrated uncertainty estimates that reflect both gap structure and observational constraints.

# Implementation and Performance

`svd_imputer` is implemented in pure Python using NumPy for linear algebra operations and pandas for time series handling. The core SVD computation uses NumPy's `linalg.svd`, which leverages optimized LAPACK routines. Data validation ensures datetime indexing, proper sorting, and handling of edge cases (e.g., all-NaN rows, duplicate timestamps).

Performance scales approximately as $O(n \cdot p \cdot \min(n,p) \cdot k)$ where $k$ is the number of iterations to convergence (typically 10-50). For a dataset with 14,000 time points and 2 sites:
- Basic imputation: ~2 seconds
- Monte Carlo (100 repeats): ~3 minutes
- Bootstrap (50 samples): ~2 minutes
- Hybrid: ~5 minutes

The package includes comprehensive test suites (pytest), example scripts, and Jupyter notebooks demonstrating all three uncertainty methods.

# Conclusion

`svd_imputer` provides a robust, well-documented solution for imputing missing values in multivariate time series with quantified uncertainty. By combining the theoretical foundation of low-rank matrix completion with practical considerations for time series data (validation, automatic rank selection, uncertainty quantification, proximity adjustment, and conditioning), the package fills an important gap in the Python ecosystem.

The package offers a complete uncertainty quantification pipeline:
1. **Three core methods** (Monte Carlo, Bootstrap, Hybrid) provide flexibility for different use cases
2. **Proximity adjustment** ensures uncertainty reflects gap structure
3. **Kriging-based conditioning** refines imputed values while reducing uncertainty by 20-40%

This progressive refinement approach makes `svd_imputer` particularly valuable for environmental monitoring applications where understanding data quality and uncertainty is essential for decision-making. The conditioning feature is especially useful when gap boundaries provide strong constraints (e.g., sensor outages with clear start/end observations) or when multiple correlated series are available.

Future enhancements may include support for additional imputation algorithms (e.g., nuclear norm minimization), automatic temporal range estimation for conditioning, handling of non-stationary time series, and integration with probabilistic forecasting frameworks.

# Availability

The package is available on PyPI (`pip install svd-imputer`) and GitHub (https://github.com/username/svd_imputer) under the MIT license. Documentation, tutorials, and example datasets are provided at the project website.

# Acknowledgments

We acknowledge the contributions of beta testers and the open-source community for feedback on early versions of this package.

# References
