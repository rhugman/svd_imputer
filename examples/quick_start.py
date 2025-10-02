"""
Quick Start Guide for svd_imputer
==================================

This guide shows the most common use cases.
"""

# =============================================================================
# INSTALLATION
# =============================================================================

# From the package directory:
# pip install -e .

# Or if you have a released version:
# pip install svd-imputer


# =============================================================================
# BASIC USAGE
# =============================================================================

import pandas as pd
import numpy as np
from svd_imputer import Imputer

# 1. Load your data with datetime index
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# 2. Create imputer (automatic rank estimation - 95% variance)
imputer = Imputer()

# 3. Impute missing values
df_imputed = imputer.fit_transform(df)

# Done! Your data is now complete.


# =============================================================================
# CUSTOMIZATION
# =============================================================================

# Adjust variance threshold (80%, 90%, 99%, etc.)
imputer = Imputer(variance_threshold=0.90)

# Use fixed rank instead of automatic
imputer = Imputer(rank=2)

# Add data normalization
from sklearn.preprocessing import StandardScaler
imputer = Imputer(variance_threshold=0.95, scaler=StandardScaler())

# Adjust convergence parameters
imputer = Imputer(max_iters=1000, tol=1e-5)

# Suppress output
imputer = Imputer(verbose=False)


# =============================================================================
# SEPARATE FIT AND TRANSFORM
# =============================================================================

# Fit on one dataset, apply to another
imputer = Imputer()
imputer.fit(df_train)
df_test_imputed = imputer.transform(df_test)


# =============================================================================
# WORKING WITH MULTIPLE CSV FILES
# =============================================================================

import os

# Load multiple files into a single DataFrame
data_dict = {}
for filename in os.listdir('data'):
    if filename.endswith('.csv'):
        df = pd.read_csv(f'data/{filename}', index_col=0, parse_dates=True)
        series_name = filename.replace('.csv', '')
        data_dict[series_name] = df.squeeze()

# Combine into single DataFrame
df_combined = pd.DataFrame(data_dict)
df_combined = df_combined.sort_index()

# Impute
imputer = Imputer()
df_imputed = imputer.fit_transform(df_combined)


# =============================================================================
# RESAMPLING HIGH-FREQUENCY DATA
# =============================================================================

# If your data is high-frequency (e.g., hourly), resample to daily
df_daily = df.resample('D').mean()
df_daily = df_daily.dropna(how='all')

# Then impute
imputer = Imputer()
df_imputed = imputer.fit_transform(df_daily)


# =============================================================================
# FILLING DATE GAPS
# =============================================================================

# After imputation, fill in missing dates
date_range = pd.date_range(
    start=df_imputed.index.min(),
    end=df_imputed.index.max(),
    freq='D'
)

df_complete = pd.DataFrame(index=date_range, columns=df_imputed.columns)
df_complete.loc[df_imputed.index, :] = df_imputed.values

# Impute the new date gaps
imputer2 = Imputer(rank=imputer.rank_)  # Use same rank
df_complete = imputer2.fit_transform(df_complete)


# =============================================================================
# ERROR HANDLING
# =============================================================================

from svd_imputer import Imputer

try:
    imputer = Imputer()
    df_imputed = imputer.fit_transform(df)
except ValueError as e:
    print(f"Validation error: {e}")
    # Common issues:
    # - Index is not DatetimeIndex
    # - Index is not sorted
    # - All rows are NaN
    # - Insufficient data

except RuntimeError as e:
    print(f"Runtime error: {e}")
    # Imputer was not fitted before transform


# =============================================================================
# VISUALIZATION
# =============================================================================

import matplotlib.pyplot as plt

# Plot original vs imputed for one column
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_imputed.index, df_imputed['column_name'], 
        label='Imputed', color='blue', linewidth=2)
ax.scatter(df.index, df['column_name'], 
           label='Original', color='red', s=10)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()


# =============================================================================
# SAVING RESULTS
# =============================================================================

# Save imputed data
df_imputed.to_csv('imputed_data.csv')

# Save with metadata
with open('imputation_info.txt', 'w') as f:
    f.write(f"Rank used: {imputer.rank_}\n")
    f.write(f"Variance threshold: {imputer.variance_threshold}\n")
    f.write(f"Original missing: {df.isna().sum().sum()}\n")
    f.write(f"After imputation: {df_imputed.isna().sum().sum()}\n")


# =============================================================================
# COMMON PATTERNS
# =============================================================================

# Pattern 1: Simple workflow
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
df_imputed = Imputer().fit_transform(df)
df_imputed.to_csv('imputed.csv')

# Pattern 2: With resampling
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
df_daily = df.resample('D').mean().dropna(how='all')
df_imputed = Imputer(variance_threshold=0.90).fit_transform(df_daily)

# Pattern 3: Multiple files + imputation
files = ['site1.csv', 'site2.csv', 'site3.csv']
data = pd.DataFrame({
    f.replace('.csv', ''): pd.read_csv(f, index_col=0, parse_dates=True).squeeze()
    for f in files
})
data = data.sort_index()
data_imputed = Imputer().fit_transform(data)
