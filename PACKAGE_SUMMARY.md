# SVD Time Series Imputer - Package Summary

## 📦 Package Structure

```
svd_imputer/
├── __init__.py           # Package initialization, exports Imputer class
├── preprocessing.py      # Data validation functions
└── imputer.py           # Core SVD imputation logic and Imputer class
```

## 🎯 Key Features

### 1. **Data Validation** (`preprocessing.py`)
- ✅ Checks datetime index exists and is properly formatted
- ✅ Verifies index is sorted and monotonically increasing
- ✅ Checks for duplicate timestamps
- ✅ Removes all-NaN rows automatically
- ✅ Validates sufficient data for imputation
- ✅ Warns if data is very sparse (>80% missing)

### 2. **Automatic Rank Estimation** (`imputer.py`)
- ✅ Uses SVD to analyze variance explained
- ✅ Selects rank based on cumulative variance threshold (default: 95%)
- ✅ Ensures optimal balance between accuracy and model complexity

### 3. **Core SVD Imputation Algorithm** (`imputer.py`)
- ✅ Iterative SVD-based imputation
- ✅ Initializes missing values with column means
- ✅ Converges to low-rank approximation
- ✅ Preserves observed values
- ✅ Optional data normalization via StandardScaler
- ✅ Convergence warnings and error handling

### 4. **Scikit-learn Style API** (`imputer.py`)
- ✅ `fit()` - Estimate rank and prepare imputer
- ✅ `transform()` - Impute missing values
- ✅ `fit_transform()` - Combined fit and transform
- ✅ `get_params()` / `set_params()` - Parameter management

## 📋 API Reference

### Class: `Imputer`

```python
from svd_imputer import Imputer

imputer = Imputer(
    variance_threshold=0.95,  # Variance to preserve (0-1)
    rank=None,                # Fixed rank (None = automatic)
    max_iters=500,            # Max SVD iterations
    tol=1e-4,                 # Convergence tolerance
    scaler=None,              # Optional StandardScaler
    verbose=True              # Print progress
)
```

### Methods

**`fit(X)`**
- Validates data and estimates rank
- Parameters: `X` (DataFrame with datetime index)
- Returns: `self`

**`transform(X)`**
- Imputes missing values
- Parameters: `X` (DataFrame with datetime index)
- Returns: Imputed DataFrame

**`fit_transform(X)`**
- Combined fit and transform
- Parameters: `X` (DataFrame with datetime index)
- Returns: Imputed DataFrame

## 🚀 Quick Start

```python
import pandas as pd
from svd_imputer import Imputer

# Load data
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Impute
imputer = Imputer()
df_imputed = imputer.fit_transform(df)

# Save
df_imputed.to_csv('imputed_data.csv')
```

## 📊 Use Cases

### 1. Single File Imputation
```python
df = pd.read_csv('timeseries.csv', index_col=0, parse_dates=True)
df_imputed = Imputer().fit_transform(df)
```

### 2. Multiple Files
```python
import os
data = pd.DataFrame({
    f.replace('.csv', ''): pd.read_csv(f'data/{f}', 
                                       index_col=0, 
                                       parse_dates=True).squeeze()
    for f in os.listdir('data') if f.endswith('.csv')
})
data_imputed = Imputer().fit_transform(data.sort_index())
```

### 3. High-Frequency Data (with Resampling)
```python
df = pd.read_csv('hourly_data.csv', index_col=0, parse_dates=True)
df_daily = df.resample('D').mean().dropna(how='all')
df_imputed = Imputer().fit_transform(df_daily)
```

### 4. Custom Parameters
```python
imputer = Imputer(
    variance_threshold=0.90,  # 90% variance
    max_iters=1000,           # More iterations
    tol=1e-5                  # Stricter convergence
)
df_imputed = imputer.fit_transform(df)
```

### 5. With Normalization
```python
from sklearn.preprocessing import StandardScaler

imputer = Imputer(scaler=StandardScaler())
df_imputed = imputer.fit_transform(df)
```

## 🔧 Installation

```bash
# From package directory
pip install -e .

# Or with requirements
pip install -r requirements.txt
```

## 📁 Example Files

1. **`basic_usage.py`** - 5 basic examples demonstrating all features
2. **`advanced_usage.py`** - Real-world workflow with CSV files
3. **`quick_start.py`** - Quick reference guide with code snippets
4. **`test_package.py`** - Comprehensive test suite

## ✅ Test Results

All 6 tests passed:
- ✓ Basic imputation
- ✓ Automatic rank estimation
- ✓ Fixed rank usage
- ✓ Data validation
- ✓ Separate fit/transform
- ✓ StandardScaler integration

## 🎓 Technical Details

### Algorithm
1. **Initialization**: Fill missing values with column means
2. **Iteration**: 
   - Compute SVD: `X = U Σ V^T`
   - Low-rank approximation: `X_approx = U[:, :r] Σ[:r] V^T[:r, :]`
   - Update missing entries only
3. **Convergence**: Stop when relative change < tolerance

### Rank Estimation
- Compute SVD on initially-imputed data
- Calculate cumulative variance explained: `Σ(σ_i²) / Σ(σ²)`
- Select minimum rank where cumulative variance ≥ threshold

### Validation
- DatetimeIndex required
- Must be sorted and unique
- Removes all-NaN rows
- Warns if >80% missing

## 📝 Dependencies

- `numpy >= 1.20.0` - Numerical computations and SVD
- `pandas >= 1.3.0` - DataFrame handling and time series
- `scikit-learn >= 1.0.0` - StandardScaler (optional)

## 🔍 Differences from `utils.py`

### Improvements Made:
1. **Better error handling** - Validates rank, checks convergence
2. **Comprehensive validation** - DateTime checks, sorting, duplicates
3. **Automatic rank estimation** - Based on variance explained
4. **Object-oriented API** - Scikit-learn style
5. **Better scaler integration** - Applied after initial imputation
6. **Warnings** - For convergence issues and sparse data
7. **Documentation** - Extensive docstrings and examples
8. **Type hints** - For better IDE support

## 🎯 Next Steps (Optional)

1. **Add unit tests** in `tests/` directory
2. **Add validation metrics** (RMSE, MAE for holdout sets)
3. **Add plotting utilities** for visualization
4. **Add support for other scalers** (MinMaxScaler, RobustScaler)
5. **Add parallel processing** for large datasets
6. **Add progress bars** for long computations
7. **Add configuration file** support
8. **Publish to PyPI** for `pip install svd-imputer`

## 📞 Usage Support

Run examples to see the package in action:
```bash
python examples/basic_usage.py      # All basic features
python examples/advanced_usage.py   # Real-world workflow
python examples/test_package.py     # Run tests
```

For quick reference, see `examples/quick_start.py`
