"""
Data validation and preprocessing utilities for time series imputation.

This module contains functions to validate and prepare time series data
for SVD-based imputation.
"""

import pandas as pd
import numpy as np
from typing import Union


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare a DataFrame for SVD imputation.
    
    Performs the following checks and operations:
    1. Ensures the index is a DatetimeIndex
    2. Verifies the index is sorted and monotonically increasing
    3. Removes rows that are all NaN
    4. Validates there is sufficient data for imputation
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data. Must have a datetime index.
        
    Returns
    -------
    pd.DataFrame
        Validated and cleaned DataFrame ready for imputation.
        
    Raises
    ------
    ValueError
        If validation fails (e.g., non-datetime index, unsorted index, 
        insufficient data, all NaN rows).
    TypeError
        If input is not a pandas DataFrame.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4],
    ...     'B': [np.nan, 2, 3, 4]
    ... }, index=pd.date_range('2020-01-01', periods=4))
    >>> validated_df = validate_dataframe(df)
    """
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(df).__name__}"
        )
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame index must be a DatetimeIndex. "
            "Use pd.to_datetime() to convert your index, or set parse_dates=True "
            "when reading from CSV."
        )
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove rows that are all NaN
    n_rows_before = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    n_rows_after = len(df_clean)
    
    if n_rows_after == 0:
        raise ValueError("All rows contain only NaN values")
    
    if n_rows_before > n_rows_after:
        n_removed = n_rows_before - n_rows_after
        print(f"Removed {n_removed} row(s) with all NaN values")
    
    # Check if index is sorted
    if not df_clean.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be sorted in increasing order. "
            "Use df.sort_index() to sort your data."
        )
    
    # Check for duplicate indices
    if df_clean.index.duplicated().any():
        n_duplicates = df_clean.index.duplicated().sum()
        raise ValueError(
            f"DataFrame index contains {n_duplicates} duplicate timestamp(s). "
            "Each timestamp must be unique."
        )
    
    # Validate sufficient data for imputation
    n_rows, n_cols = df_clean.shape
    
    if n_rows < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 rows, got {n_rows}"
        )
    
    if n_cols < 1:
        raise ValueError(
            f"Insufficient data: need at least 1 column, got {n_cols}"
        )
    
    # Check if there's any non-NaN data
    if df_clean.isna().all().all():
        raise ValueError("All values in DataFrame are NaN")
    
    # Warn if data is very sparse
    nan_percentage = (df_clean.isna().sum().sum() / (n_rows * n_cols)) * 100
    if nan_percentage > 80:
        print(
            f"Warning: Data is {nan_percentage:.1f}% missing. "
            "SVD imputation may not perform well with very sparse data."
        )
    
    return df_clean


def check_sufficient_rank(df: pd.DataFrame, min_rank: int = 1) -> None:
    """
    Check if the data has sufficient dimensions for the requested rank.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_rank : int, optional
        Minimum rank required (default: 1)
        
    Raises
    ------
    ValueError
        If the data dimensions are insufficient for the requested rank
    """
    n_rows, n_cols = df.shape
    max_possible_rank = min(n_rows, n_cols)
    
    if min_rank > max_possible_rank:
        raise ValueError(
            f"Requested rank {min_rank} exceeds maximum possible rank "
            f"{max_possible_rank} for data with shape {df.shape}"
        )
    
    if max_possible_rank < 1:
        raise ValueError(
            f"Data dimensions {df.shape} are insufficient for SVD imputation"
        )



def detrend_timeseries(X):
    """Detrend each column (time series)"""
    X_detrended = np.copy(X)
    trends = np.zeros_like(X)
    
    for col in range(X.shape[1]):
        mask = ~np.isnan(X[:, col])
        if mask.sum() > 2:  # Need at least 2 points
            # Fit trend on observed values only
            t = np.arange(len(X))[mask]
            values = X[mask, col]
            trend_coef = np.polyfit(t, values, deg=1)
            
            # Store trend for all time points
            trends[:, col] = np.polyval(trend_coef, np.arange(len(X)))
            X_detrended[:, col] = X[:, col] - trends[:, col]
    
    return X_detrended, trends

def standardize_columns(X):
    """Standardize each column to mean=0, std=1"""
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    X_standardized = (X - means) / stds
    return X_standardized, means, stds


def preprocess_for_svd(data):
    """Full preprocessing pipeline"""
    if isinstance(data, pd.DataFrame):
        X = data.copy().values.astype(float)
    elif isinstance(data, np.ndarray):
        X = data.copy().astype(float)
    # 1. Detrend (if non-stationary)
    X_detrended, trends = detrend_timeseries(X)
    
    # 2. Standardize (always good)
    X_std, means, stds = standardize_columns(X)
    
    # 3. Fill with mean (now mean=0 after standardization)
    X_filled = np.where(np.isnan(X_std), 0, X_std)
    if isinstance(data, pd.DataFrame):
        X_filled = pd.DataFrame(X_filled,columns=data.columns,index=data.index)
    
    return X_filled, (trends, means, stds)

def postprocess_after_svd(X_imputed, preprocessing_info):
    """Reverse preprocessing"""
    trends, means, stds = preprocessing_info
    
    # 1. Unstandardize
    X = X_imputed * stds + means
    
    # 2. Add trends back
    #X = X + trends
    
    return X
