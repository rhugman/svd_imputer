# SVD Time Series Imputer

[![PyPI version](https://badge.fury.io/py/svd-imputer.svg)](https://badge.fury.io/py/svd-imputer)
[![Python versions](https://img.shields.io/pypi/pyversions/svd-imputer.svg)](https://pypi.org/project/svd-imputer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for time series imputation using Singular Value Decomposition (SVD) with automatic rank estimation and uncertainty quantification.

## Table of Contents
- [Installation](#installation)  
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Requirements](#requirements)

A Python package for time series imputation using SVD with automatic rank estimation, uncertainty quantification, and scikit-learn compatible API.

## Installation

**PyPI (Recommended)**:
```bash
pip install svd-imputer
```

**From Source** (development version):
```bash
git clone https://github.com/rhugman/svd_imputer.git
cd svd_imputer
pip install -e .
```

**With Development Dependencies**:
```bash
pip install -e ".[dev]"
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

# With uncertainty estimation (Multiple Imputation)
df_imputed, df_uncertainty = imputer.fit_transform(return_uncertainty=True, n_imputations=10)
print(f"Average uncertainty: {df_uncertainty.mean().mean():.3f}")
```

> **Note**: The `Imputer` class uses a data-centric design where data is provided at initialization and preprocessed once. This ensures consistency across all analyses and eliminates redundant preprocessing operations.

## Usage

```python
from svd_imputer import Imputer

# Basic imputation (automatic rank estimation)
imputer = Imputer(data=df, variance_threshold=0.95)
df_imputed = imputer.fit_transform()

# Cross-validation optimization
imputer = Imputer(data=df, rank="auto")
imputer.fit()
print(f"Optimized rank: {imputer.rank_}")

# With uncertainty estimation (Multiple Imputation)
df_imputed, df_uncertainty = imputer.fit_transform(return_uncertainty=True, n_imputations=10)
# df_uncertainty contains standard deviations for each imputed value

# Advanced: model diagnostics
residuals, stats = imputer.calculate_reconstruction_residuals(return_stats=True)
print(f"Reconstruction R²: {stats['r_squared']:.3f}")
```

## Configuration

```python
imputer = Imputer(
    data=df,                    # Input DataFrame (required)
    variance_threshold=0.95,    # Variance threshold for auto rank estimation
    rank=None,                  # None (auto-estimate), int (fixed), or "auto" (optimize)
    max_iters=500,             # Maximum SVD iterations
    tol=1e-4,                  # Convergence tolerance  
    verbose=True               # Enable logging output
)
```


## Examples

Complete examples are available in the `examples/` directory:
- `basic_example.ipynb` - Basic usage and quick start tutorial
- `augmented_example.ipynb` - Extended examples with data agumentation features

## How It Works

The algorithm performs iterative SVD imputation with automatic rank estimation:

1. **Preprocessing**: Data validation, standardization, and missing value handling
2. **Rank Estimation**: Variance threshold, cross-validation, or fixed rank
3. **SVD Imputation**: Iterative low-rank approximation until convergence
4. **Uncertainty Estimation**: Multiple Imputation (Stochastic SVD) or Monte Carlo validation

## API Reference

### Main Class
`Imputer(data, variance_threshold=0.95, rank=None, max_iters=500, tol=1e-4, verbose=True)`

### Key Methods
- `fit()` / `transform()` / `fit_transform()`: Standard sklearn interface
- `estimate_uncertainty()`: Monte Carlo validation
- `calculate_reconstruction_residuals()`: Model diagnostics
- `project_data()` / `reconstruct_data()`: SVD subspace operations

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## Performance Notes

- **Memory**: O(n × m) for data size n×m, plus O(min(n,m)²) for SVD decomposition
- **Time Complexity**: O(k × min(n,m)³) where k is the number of SVD iterations  
- **Recommended Scale**: Efficient for datasets up to ~10,000 × 100 dimensions
- **Optimization**: SVD components are cached for efficient reuse across operations

## Package Status

**Current Status**: **Published on PyPI** 🎉

This package is currently in **Beta** - the core functionality is stable and tested (86 tests passing), but the API may evolve. Suitable for research and development use.

## Disclaimer

**IMPORTANT**: This software is provided "as is" without warranty of any kind. The authors and contributors make no representations or warranties regarding the accuracy, completeness, or validity of the code or its results. Users are solely responsible for validating the appropriateness and correctness of this software for their specific use cases. The authors assume no responsibility or liability for any errors, omissions, or damages arising from the use of this software.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **PyPI Package**: https://pypi.org/project/svd-imputer/
- **Source Code**: https://github.com/rhugman/svd_imputer
- **Issues**: https://github.com/rhugman/svd_imputer/issues

## Citation

If you use this package in your research, please cite:

```bibtex
@software{svd_time_series_imputer,
  title={SVD Time Series Imputer: A Python Package for Missing Data Imputation},
  author={Rui Hugman},
  year={2025},
  url={https://github.com/rhugman/svd_imputer},
  note={Available on PyPI: https://pypi.org/project/svd-imputer/}
}
```
