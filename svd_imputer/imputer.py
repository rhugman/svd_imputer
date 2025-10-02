"""
SVD-based time series imputation with automatic rank estimation.

This module contains the core SVD imputation algorithm, rank estimation,
and the main Imputer class.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler

from .preprocessing import validate_dataframe, check_sufficient_rank


def estimate_rank(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Estimate optimal rank based on cumulative variance explained.
    
    Performs SVD on the data (with NaN values temporarily filled) and 
    determines the rank needed to capture the specified variance threshold.
    
    Parameters
    ----------
    X : np.ndarray
        Input data array (may contain NaN values)
    variance_threshold : float, optional
        Fraction of variance to preserve (default: 0.95 for 95%)
        
    Returns
    -------
    int
        Estimated optimal rank
        
    Examples
    --------
    >>> X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    >>> rank = estimate_rank(X, variance_threshold=0.95)
    """
    # Temporarily fill NaN with column means for rank estimation
    X_temp = X.copy()
    col_means = np.nanmean(X_temp, axis=0)
    inds = np.where(np.isnan(X_temp))
    X_temp[inds] = np.take(col_means, inds[1])
    
    # Perform SVD
    try:
        _, s, _ = np.linalg.svd(X_temp, full_matrices=False)
    except np.linalg.LinAlgError:
        warnings.warn(
            "SVD computation failed. Using rank=1 as fallback.",
            RuntimeWarning
        )
        return 1
    
    # Calculate cumulative variance explained
    variance_explained = (s ** 2) / np.sum(s ** 2)
    cumulative_variance = np.cumsum(variance_explained)
    
    # Find minimum rank that meets threshold
    rank = np.searchsorted(cumulative_variance, variance_threshold) + 1
    
    # Ensure rank is at least 1 and at most min(n_rows, n_cols)
    max_rank = len(s)
    rank = max(1, min(rank, max_rank))
    
    return int(rank)


def iterative_svd_impute(
    X: np.ndarray,
    rank: int = 2,
    max_iters: int = 500,
    tol: float = 1e-4,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Impute missing values using iterative SVD.
    
    This algorithm iteratively:
    1. Fills missing values with column means (initialization)
    2. Computes SVD and low-rank approximation
    3. Updates missing values with approximation
    4. Repeats until convergence
    
    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    rank : int, optional
        Number of singular values/vectors to keep (default: 2)
    max_iters : int, optional
        Maximum number of iterations (default: 500)
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    scaler : StandardScaler, optional
        Optional scaler to normalize data before imputation
        
    Returns
    -------
    np.ndarray
        Array with imputed values
        
    Raises
    ------
    ValueError
        If rank exceeds matrix dimensions
        
    Examples
    --------
    >>> X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> X_imputed = iterative_svd_impute(X, rank=2)
    """
    # Validate rank
    n_rows, n_cols = X.shape
    max_possible_rank = min(n_rows, n_cols)
    if rank > max_possible_rank:
        raise ValueError(
            f"rank={rank} exceeds maximum possible rank {max_possible_rank} "
            f"for matrix with shape {X.shape}"
        )
    
    X_filled = X.copy()
    
    # Apply scaling if provided
    if scaler is not None:
        # Note: StandardScaler.fit_transform doesn't handle NaN well
        # So we only scale after initial imputation
        pass
    
    # Step 1: Initialize missing values with column means
    col_means = np.nanmean(X_filled, axis=0)
    inds = np.where(np.isnan(X_filled))
    X_filled[inds] = np.take(col_means, inds[1])
    
    # Apply scaling after initial imputation if scaler provided
    if scaler is not None:
        X_filled = scaler.fit_transform(X_filled)
    
    # Step 2: Iterative updates
    converged = False
    for it in range(max_iters):
        # Compute SVD
        try:
            U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"SVD failed at iteration {it}. Returning current state.",
                RuntimeWarning
            )
            break
        
        # Low-rank approximation
        S = np.diag(s[:rank])
        X_approx = U[:, :rank] @ S @ Vt[:rank, :]
        
        # Update only the originally missing entries
        X_new = X_filled.copy()
        X_new[inds] = X_approx[inds]
        
        # Check convergence
        diff = np.linalg.norm(X_new - X_filled) / np.linalg.norm(X_filled)
        X_filled = X_new
        
        if diff < tol:
            converged = True
            break
    
    # Warn if didn't converge
    if not converged:
        warnings.warn(
            f"Max iterations ({max_iters}) reached without convergence "
            f"(diff={diff:.2e}). Consider increasing max_iters or relaxing tol.",
            RuntimeWarning
        )
    
    # Inverse transform if scaler was used
    if scaler is not None:
        X_filled = scaler.inverse_transform(X_filled)
    
    return X_filled


class Imputer:
    """
    SVD-based time series imputer with automatic rank estimation.
    
    This class provides a scikit-learn style interface for imputing missing
    values in time series data using Singular Value Decomposition (SVD).
    
    Parameters
    ----------
    variance_threshold : float, optional
        Fraction of variance to preserve for automatic rank estimation.
        Default is 0.95 (95% of variance). Must be between 0 and 1.
    rank : int, optional
        Fixed rank to use. If None (default), rank is estimated automatically
        based on variance_threshold.
    max_iters : int, optional
        Maximum number of SVD iterations (default: 500)
    tol : float, optional
        Convergence tolerance (default: 1e-4)
    scaler : StandardScaler or None, optional
        Optional scaler to normalize data before imputation (default: None)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Attributes
    ----------
    rank_ : int
        The rank used for imputation (set after fitting)
    is_fitted_ : bool
        Whether the imputer has been fitted
    columns_ : list
        Column names from the fitted DataFrame
    index_name_ : str
        Name of the index from the fitted DataFrame
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from svd_imputer import Imputer
    >>> 
    >>> # Create sample data
    >>> dates = pd.date_range('2020-01-01', periods=10)
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
    ...     'B': [10, np.nan, 30, 40, np.nan, 60, 70, 80, 90, 100]
    ... }, index=dates)
    >>> 
    >>> # Automatic rank estimation
    >>> imputer = Imputer(variance_threshold=0.95)
    >>> df_imputed = imputer.fit_transform(df)
    >>> 
    >>> # Fixed rank
    >>> imputer = Imputer(rank=2)
    >>> df_imputed = imputer.fit_transform(df)
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.95,
        rank: Optional[int] = None,
        max_iters: int = 500,
        tol: float = 1e-4,
        scaler: Optional[StandardScaler] = None,
        verbose: bool = True
    ):
        # Validate parameters
        if variance_threshold <= 0 or variance_threshold > 1:
            raise ValueError(
                f"variance_threshold must be between 0 and 1, got {variance_threshold}"
            )
        
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}")
        
        if max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {max_iters}")
        
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        
        self.variance_threshold = variance_threshold
        self.rank = rank
        self.max_iters = max_iters
        self.tol = tol
        self.scaler = scaler
        self.verbose = verbose
        
        # Attributes set during fitting
        self.rank_ = None
        self.is_fitted_ = False
        self.columns_ = None
        self.index_name_ = None
    
    def fit(self, X: pd.DataFrame) -> 'Imputer':
        """
        Fit the imputer on the data (estimate rank if needed).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
            
        Returns
        -------
        self
            Fitted imputer
        """
        # Validate input data
        X_validated = validate_dataframe(X)
        
        # Store metadata
        self.columns_ = X_validated.columns.tolist()
        self.index_name_ = X_validated.index.name
        
        # Convert to numpy array
        X_array = X_validated.values
        
        # Estimate rank if not provided
        if self.rank is None:
            self.rank_ = estimate_rank(X_array, self.variance_threshold)
            if self.verbose:
                print(f"Estimated rank: {self.rank_} "
                      f"(variance threshold: {self.variance_threshold*100:.0f}%)")
        else:
            self.rank_ = self.rank
            # Check if requested rank is feasible
            check_sufficient_rank(X_validated, self.rank_)
            if self.verbose:
                print(f"Using fixed rank: {self.rank_}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
            
        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values
        """
        # Check if fitted
        if not self.is_fitted_:
            raise RuntimeError(
                "Imputer must be fitted before transform. "
                "Call fit() or use fit_transform()."
            )
        
        # Validate input data
        X_validated = validate_dataframe(X)
        
        # Convert to numpy array
        X_array = X_validated.values
        
        # Perform imputation
        if self.verbose:
            print(f"Imputing with rank={self.rank_}, max_iters={self.max_iters}, tol={self.tol}")
        
        X_imputed = iterative_svd_impute(
            X_array,
            rank=self.rank_,
            max_iters=self.max_iters,
            tol=self.tol,
            scaler=self.scaler
        )
        
        # Convert back to DataFrame
        df_imputed = pd.DataFrame(
            X_imputed,
            index=X_validated.index,
            columns=X_validated.columns
        )
        
        if self.verbose:
            n_imputed = np.isnan(X_array).sum()
            print(f"Imputed {n_imputed} missing value(s)")
        
        return df_imputed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform the data in one step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
            
        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values
        """
        return self.fit(X).transform(X)
    
    def get_params(self) -> dict:
        """
        Get parameters for this estimator.
        
        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        return {
            'variance_threshold': self.variance_threshold,
            'rank': self.rank,
            'max_iters': self.max_iters,
            'tol': self.tol,
            'scaler': self.scaler,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'Imputer':
        """
        Set parameters for this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
