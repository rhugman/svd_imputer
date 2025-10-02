# SVD Time Series Imputer

A simple and streamlined Python package for time series imputation using Singular Value Decomposition (SVD) with automatic rank estimation.

## Features

- **Automatic rank estimation** based on variance energy threshold (default: 95%)
- **Data validation** ensures proper datetime index, ordering, and data quality
- **Scikit-learn style API** with `fit()`, `transform()`, and `fit_transform()` methods
- **Minimal dependencies** (numpy, pandas, scikit-learn)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd
from svd_imputer import Imputer

# Load your time series data (with datetime index)
df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Create imputer with automatic rank estimation
imputer = Imputer(variance_threshold=0.95)

# Fit and transform
df_imputed = imputer.fit_transform(df)
```

## Usage

### Basic Example

```python
from svd_imputer import Imputer

# Initialize with default settings (95% variance threshold)
imputer = Imputer()

# Fit on your data and transform
df_imputed = imputer.fit_transform(df)

# Or use fit/transform separately
imputer.fit(df)
df_imputed = imputer.transform(df)
```

### Custom Parameters

```python
# Specify variance threshold for rank estimation
imputer = Imputer(
    variance_threshold=0.90,  # Use 90% variance explained
    max_iters=500,            # Maximum SVD iterations
    tol=1e-4,                 # Convergence tolerance
    scaler=None               # Optional: StandardScaler() for normalization
)
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## How It Works

1. **Data Validation**: Checks datetime index, ordering, and removes invalid rows
2. **Rank Estimation**: Computes optimal rank based on variance energy threshold
3. **SVD Imputation**: Iteratively imputes missing values using low-rank SVD approximation

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
