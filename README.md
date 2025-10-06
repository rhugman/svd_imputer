# SVD Time Series Imputer

A simple and efficient Python package for time series imputation using Singular Value Decomposition (SVD) wand imputation error/uncertainty quantification.

## Table of Contents
- [Features](#features)
- [Installation](#installation)  
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Requirements](#requirements)
- [Performance Notes](#performance-notes)

## Features

### Core Functionality
- **Multiple rank estimation methods**:
  - Automatic estimation based on variance threshold (default: 95%)
  - Cross-validation optimization (`rank="auto"`)
  - Fixed rank specification
- **Robust SVD imputation** with iterative convergence and automatic fallbacks
- **Uncertainty estimation** using Monte Carlo validation
- **Internal Data preprocessing** i.e., standardization
- **Scikit-learn compatible API** with `fit()`, `transform()`, and `fit_transform()` methods

### Advanced Features  
- **Model diagnostics**:
  - Reconstruction residual analysis
  - Rank optimization with cross-validation
  - SVD component analysis and projection
- **Custom masking strategies** for Monte Carlo validation:
  - Random masking with row integrity preservation
  - Block temporal masking with automatic fallbacks
- **Minimal dependencies** (numpy, pandas, scikit-learn)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd
import numpy as np
from svd_imputer import Imputer

# Load your time series data (with datetime index)
df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Simple imputation with automatic rank estimation
imputer = Imputer(data=df, variance_threshold=0.95)
df_imputed = imputer.fit_transform()

# With uncertainty estimation  
df_imputed, uncertainty = imputer.fit_transform(return_uncertainty=True)
print(f"RMSE: {uncertainty['rmse']:.3f} ± {uncertainty['rmse_std']:.3f}")
```

> **Note**: The `Imputer` class follows a data-centric design where data is provided at initialization and preprocessed once. This eliminates redundant operations and ensures consistency across all analyses.

## Usage

### Basic Examples

```python
from svd_imputer import Imputer

# 1. Basic imputation with automatic rank estimation (95% variance)
imputer = Imputer(data=df, variance_threshold=0.95)
df_imputed = imputer.fit_transform()

# 2. Using cross-validation to optimize rank
imputer = Imputer(data=df, rank="auto")
df_imputed = imputer.fit_transform()
print(f"Optimized rank: {imputer.rank_}")

# 3. Fixed rank specification
imputer = Imputer(data=df, rank=3)
df_imputed = imputer.fit_transform()

# 4. Separate fit/transform for multiple datasets
imputer = Imputer(data=df_train, rank=2)
imputer.fit()
df_train_imputed = imputer.transform()
df_test_imputed = imputer.transform()  # Apply same model to new data
```

### Uncertainty Quantification

```python
# Monte Carlo uncertainty estimation
imputer = Imputer(data=df, variance_threshold=0.95)
df_imputed, uncertainty = imputer.fit_transform(
    return_uncertainty=True,
    n_repeats=100,
    mask_strategy='random'  # or 'block' for temporal blocks
)

# Access uncertainty metrics
print(f"RMSE: {uncertainty['rmse']:.3f} ± {uncertainty['rmse_std']:.3f}")
print(f"MAE: {uncertainty['mae']:.3f}")
print(f"95% CI: {uncertainty['rmse_ci']}")
```

### Advanced Features

```python
# Model diagnostics and residual analysis
imputer = Imputer(data=df, rank=3)
imputer.fit()
df_imputed = imputer.transform()

# Calculate reconstruction residuals
residuals, stats = imputer.calculate_reconstruction_residuals(
    return_stats=True
)
print(f"Reconstruction R²: {stats['r_squared']:.3f}")
print(f"RMSE: {stats['rmse']:.4f}")

# Project new data onto learned SVD subspace
df_projected = imputer.project_data(new_df)

# Reconstruct data for denoising/compression
df_reconstructed = imputer.reconstruct_data()
```

### Rank Optimization

```python
# Comprehensive rank optimization
imputer = Imputer(data=df)
results = imputer.optimize_rank(
    rank_range=(1, 10),
    cv_folds=5,
    n_repeats_per_fold=20,
    mask_strategy='random'
)

print(f"Optimal rank: {results['optimal_rank']}")
print(results['results_df'])  # Detailed results for all ranks
```

## Configuration

### Parameters

```python
# All available parameters
imputer = Imputer(
    data=df,                    # Input DataFrame (required)
    variance_threshold=0.95,    # Variance threshold for auto rank estimation
    rank=None,                  # None (auto-estimate), int (fixed), or "auto" (optimize)
    max_iters=500,             # Maximum SVD iterations
    tol=1e-4,                  # Convergence tolerance  
    verbose=True               # Enable logging output
)
```


## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## Examples

Complete examples are available in the `examples/` directory:
- `basic_example.ipynb` - Simple Jupyter notebook

## How It Works

### Algorithm Overview
1. **Data Validation & Preprocessing**: 
   - Validates datetime index and data quality
   - Applies detrending and standardization
   - Handles missing values appropriately

2. **Rank Estimation**:
   - **Variance-based**: Finds rank capturing specified variance threshold
   - **Cross-validation**: Tests multiple ranks to minimize imputation error  
   - **Fixed**: Uses user-specified rank

3. **Iterative SVD Imputation**:
   - Fills missing values iteratively using low-rank SVD approximation
   - Monitors convergence with configurable tolerance
   - Caches SVD components for efficiency

4. **Uncertainty Quantification**:
   - **Monte Carlo validation**: Masks observed values and measures reconstruction error
   - **Multiple masking strategies**: Random or temporal block masking

5. **Post-processing**:
   - Restores original scale and temporal trends

### Key Features
- **Custom masking**: Prevents creation of entirely missing rows during validation
- **Automatic fallbacks**: Switches strategies when block masking fails
- **SVD component caching**: Enables efficient reuse for multiple datasets

## API Reference

### Main Class
- `Imputer(data, variance_threshold, rank, max_iters, tol, verbose)`: Main imputation class

### Core Methods  
- `fit()`: Fit imputer to data and estimate rank
- `transform()`: Apply imputation to data
- `fit_transform(return_uncertainty)`: Combined fit and transform with optional uncertainty

### Analysis Methods
- `estimate_uncertainty(n_repeats, mask_strategy)`: Monte Carlo uncertainty estimation
- `optimize_rank(rank_range, cv_folds)`: Cross-validation rank optimization
- `calculate_reconstruction_residuals()`: Model diagnostic analysis
- `project_data(new_data)`: Project data onto learned SVD subspace
- `reconstruct_data()`: Reconstruct using low-rank approximation

### Utility Methods
- `get_optimization_results()`: Access detailed optimization results
- `get_params()` / `set_params()`: Scikit-learn style parameter access

## Performance Notes

- **Memory**: O(n × m) for data size n×m, plus O(min(n,m)²) for SVD
- **Time**: O(k × min(n,m)³) where k is number of iterations  
- **Scalability**: Efficient for datasets up to ~10,000 × 100 dimensions
- **Caching**: SVD components cached for fast repeated transforms

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{svd_time_series_imputer,
  title={SVD Time Series Imputer: A Python Package for Missing Data Imputation},
  author={[Author Name]},
  year={2025},
  url={https://github.com/rhugman/ranger.svdtseries}
}
```
