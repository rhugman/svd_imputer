"""
Data validation and preprocessing utilities for time series imputation.

This module contains functions to validate and prepare time series data
for SVD-based imputation.
"""

import numpy as np
import pandas as pd


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
        raise TypeError(f"Input must be a pandas DataFrame, got {type(df).__name__}")

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
    df_clean = df_clean.dropna(how="all")
    n_rows_after = len(df_clean)

    if n_rows_after == 0:
        raise ValueError("All rows contain only NaN values")

    if n_rows_before > n_rows_after:
        n_removed = n_rows_before - n_rows_after
        print(f"Removed {n_removed} row(s) with all NaN values")

    # Check if index is sorted
    if not df_clean.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in increasing order. " "Use df.sort_index() to sort your data.")

    # Check for duplicate indices
    if df_clean.index.duplicated().any():
        n_duplicates = df_clean.index.duplicated().sum()
        raise ValueError(f"DataFrame index contains {n_duplicates} duplicate timestamp(s). " "Each timestamp must be unique.")

    # Validate sufficient data for imputation
    n_rows, n_cols = df_clean.shape

    if n_rows < 2:
        raise ValueError(f"Insufficient data: need at least 2 rows, got {n_rows}")

    if n_cols < 1:
        raise ValueError(f"Insufficient data: need at least 1 column, got {n_cols}")

    # Check if there's any non-NaN data
    if df_clean.isna().all().all():
        raise ValueError("All values in DataFrame are NaN")

    # Warn if data is very sparse
    nan_percentage = (df_clean.isna().sum().sum() / (n_rows * n_cols)) * 100
    if nan_percentage > 80:
        print(f"Warning: Data is {nan_percentage:.1f}% missing. " "SVD imputation may not perform well with very sparse data.")

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
            f"Requested rank {min_rank} exceeds maximum possible rank " f"{max_possible_rank} for data with shape {df.shape}"
        )

    if max_possible_rank < 1:
        raise ValueError(f"Data dimensions {df.shape} are insufficient for SVD imputation")


def detrend_timeseries(X):
    """
    Remove linear trends from each column (time series) in a 2D array.

    For each column, fits a linear trend (using least squares) to the observed (non-NaN)
    values and subtracts this trend from the entire column. The function returns both
    the detrended data and the estimated trends.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_series), where each column is a time series.
        May contain NaN values, which are ignored when fitting the trend.

    Returns
    -------
    X_detrended : np.ndarray
        Array of the same shape as `X`, with the linear trend removed from each column.
    trends : np.ndarray
        Array of the same shape as `X`, containing the estimated linear trend for each column.

    Notes
    -----
    - Columns with fewer than 2 non-NaN values are left unchanged (trend is zero).
    - NaN values in `X` are preserved in the output.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 4], [3, 6], [np.nan, 8]])
    >>> X_detrended, trends = detrend_timeseries(X)
    """
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
    """
    Standardize each column of the input array to mean 0 and standard deviation 1.

    This function computes the mean and standard deviation of each column (ignoring NaNs),
    and returns the standardized array along with the means and standard deviations used.

    Parameters
    ----------
    X : np.ndarray
        Input 2D array (matrix) with shape (n_samples, n_features). Can contain NaN values.

    Returns
    -------
    X_standardized : np.ndarray
        Array of the same shape as X, with each column standardized (mean=0, std=1).
        For columns with zero variance (std=0), values are set to 0.
    means : np.ndarray
        1D array of means for each column (shape: n_features,).
    stds : np.ndarray
        1D array of standard deviations for each column (shape: n_features,).

    Notes
    -----
    NaN values are ignored when computing means and standard deviations, and remain in the output.
    Columns with zero variance (constant values) are standardized to zeros.
    """
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)

    # Handle division by zero for columns with no variance
    with np.errstate(divide="ignore", invalid="ignore"):
        X_standardized = (X - means) / stds

    # Set standardized values to 0 for columns with zero variance
    zero_var_mask = stds == 0
    X_standardized[:, zero_var_mask] = 0

    return X_standardized, means, stds


def preprocess_for_svd(data):
    """
    Prepare data for SVD imputation by standardizing columns.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data.

    Returns
    -------
    tuple
        (standardized_data, (means, stds))
    """
    if isinstance(data, pd.DataFrame):
        X = data.copy().values.astype(float)
    elif isinstance(data, np.ndarray):
        X = data.copy().astype(float)

    # Standardize data
    X_std, means, stds = standardize_columns(X)

    # Preserve missing values for the iterative SVD algorithm
    if isinstance(data, pd.DataFrame):
        X_std = pd.DataFrame(X_std, columns=data.columns, index=data.index)

    return X_std, (means, stds)


def postprocess_after_svd(X_imputed, preprocessing_info):
    """Reverse preprocessing"""
    means, stds = preprocessing_info

    # Unstandardize data
    X = X_imputed * stds + means

    return X


def create_derivative_augmented_matrix(df):
    """
    Augment DataFrame with first and second differences.

    Creates columns: [original, first_diff, second_diff]

    Parameters
    ----------
    df : pd.DataFrame
        Time series DataFrame with datetime index

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with original column names preserved
        Format: original cols, then _d1 suffix, then _d2 suffix
        Index is aligned to match valid rows (loses first 2 rows)
    """
    # Calculate differences
    df_d1 = df.diff()  # First difference
    df_d2 = df_d1.diff()  # Second difference

    # Align to valid rows (where second difference exists)
    # This means dropping first 2 rows
    valid_idx = df.index[2:]
    df_aligned = df.loc[valid_idx]
    df_d1_aligned = df_d1.loc[valid_idx]
    df_d2_aligned = df_d2.loc[valid_idx]

    # Rename columns to show what they represent
    df_d1_renamed = df_d1_aligned.add_suffix("_d1")
    df_d2_renamed = df_d2_aligned.add_suffix("_d2")

    # Concatenate horizontally
    df_aug = pd.concat([df_aligned, df_d1_renamed, df_d2_renamed], axis=1)

    return df_aug


def create_symmetric_augmented_matrix(df, window=1):
    """
    Include past and future lags: [X_{t-w}, ..., X_{t}, ..., X_{t+w}]

    Better for interpolation (gaps in middle of series).

    Parameters
    ----------
    df : pd.DataFrame
        Time series DataFrame with datetime index
    window : int
        Number of time steps to include before and after current time

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with lag suffixes
        Format: col_name_lag-3, col_name_lag-2, ..., col_name_lag0, ..., col_name_lag+3
        Index corresponds to "current" time (center of window)
    """
    lags = list(range(-window, window + 1))

    # Valid index range (center of windows)
    valid_idx = df.index[window:-window]

    # Build list of shifted dataframes
    df_list = []

    for lag in lags:
        if lag == 0:
            # Current time - no suffix needed for clarity, but let's be consistent
            df_shifted = df.shift(-lag).loc[valid_idx]
            df_shifted = df_shifted.add_suffix(f"_lag{lag:+d}")
        elif lag < 0:
            # Past values: shift forward (positive shift gets past values)
            df_shifted = df.shift(-lag).loc[valid_idx]
            df_shifted = df_shifted.add_suffix(f"_lag{lag:+d}")
        else:
            # Future values: shift backward (negative shift gets future values)
            df_shifted = df.shift(-lag).loc[valid_idx]
            df_shifted = df_shifted.add_suffix(f"_lag{lag:+d}")

        df_list.append(df_shifted)

    # Concatenate all lagged versions
    df_aug = pd.concat(df_list, axis=1)

    return df_aug


def create_asymmetric_augmented_matrix(df, past_lags=[1]):
    """
    Include only past lags (useful for forecasting scenarios).

    Parameters
    ----------
    df : pd.DataFrame
        Time series DataFrame with datetime index
    past_lags : list of int
        Lags to include (positive integers for past values)

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with current values and past lags
        Format: col_name_lag0, col_name_lag-1, col_name_lag-2, ...
    """
    max_lag = max(past_lags)

    # Valid index (need max_lag historical values)
    valid_idx = df.index[max_lag:]

    df_list = []

    # Current time (lag 0)
    df_current = df.loc[valid_idx].add_suffix("_lag0")
    df_list.append(df_current)

    # Past lags
    for lag in sorted(past_lags):
        df_shifted = df.shift(lag).loc[valid_idx]
        df_shifted = df_shifted.add_suffix(f"_lag-{lag}")
        df_list.append(df_shifted)

    df_aug = pd.concat(df_list, axis=1)

    return df_aug
