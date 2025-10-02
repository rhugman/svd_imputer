"""
SVD-based time series imputation with automatic rank estimation.

This module contains the core SVD imputation algorithm, rank estimation,
and the main Imputer class.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from math import sqrt
from statistics import mean, stdev

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


def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return sqrt(np.mean((true - pred) ** 2))


def _mae(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(true - pred))


def _random_mask_observed(X: np.ndarray, frac: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomly mask a fraction of observed values for validation.
    
    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    frac : float
        Fraction of observed values to mask
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = observed, False = masked)
    """
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(X)  # True = observed
    obs_indices = np.argwhere(mask)
    n_hide = int(len(obs_indices) * frac)
    hide_idx = rng.choice(len(obs_indices), size=n_hide, replace=False)
    for idx in hide_idx:
        r, c = obs_indices[idx]
        mask[r, c] = False
    return mask


def _block_mask_time(X: np.ndarray, block_len: int = 5, n_blocks: int = 1, 
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Mask temporal blocks of observed values for validation.
    
    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    block_len : int
        Length of temporal blocks to mask
    n_blocks : int
        Number of blocks to mask
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = observed, False = masked)
    """
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(X)
    n_rows, _ = X.shape
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n_rows - block_len + 1))
        rows = range(start, start + block_len)
        # Hide those rows across all columns that are observed
        for r in rows:
            for c in range(X.shape[1]):
                if not np.isnan(X[r, c]):
                    mask[r, c] = False
    return mask


def _monte_carlo_validation(
    X: np.ndarray,
    rank: int,
    scaler: Optional[StandardScaler],
    max_iters: int,
    tol: float,
    n_repeats: int = 100,
    mask_strategy: str = 'random',
    frac: float = 0.1,
    block_len: int = 5,
    n_blocks: int = 1,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform Monte Carlo validation to estimate imputation error.
    
    Parameters
    ----------
    X : np.ndarray
        Input array with np.nan for missing values
    rank : int
        Rank for SVD imputation
    scaler : StandardScaler, optional
        Optional scaler
    max_iters : int
        Maximum iterations for imputation
    tol : float
        Convergence tolerance
    n_repeats : int
        Number of Monte Carlo repeats
    mask_strategy : str
        'random' or 'block'
    frac : float
        Fraction to mask (for random strategy)
    block_len : int
        Block length (for block strategy)
    n_blocks : int
        Number of blocks (for block strategy)
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict
        Dictionary with RMSE and MAE statistics
    """
    rng = np.random.default_rng(seed)
    rmse_list = []
    mae_list = []

    for i in range(n_repeats):
        s = None if seed is None else int(rng.integers(1 << 30))
        
        # Create mask
        if mask_strategy == 'random':
            mask = _random_mask_observed(X, frac=frac, seed=s)
        elif mask_strategy == 'block':
            mask = _block_mask_time(X, block_len=block_len, n_blocks=n_blocks, seed=s)
        else:
            raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")

        # Build X_with_nans for imputation
        X_with_nans = X.copy()
        obs_all = ~np.isnan(X)
        masked_positions = np.logical_and(obs_all, ~mask)
        X_with_nans[masked_positions] = np.nan

        # Impute
        X_imputed = iterative_svd_impute(
            X_with_nans,
            rank=rank,
            max_iters=max_iters,
            tol=tol,
            scaler=scaler
        )

        # Compute error only on masked positions
        true_vals = X[masked_positions]
        pred_vals = X_imputed[masked_positions]

        rmse_list.append(_rmse(true_vals, pred_vals))
        mae_list.append(_mae(true_vals, pred_vals))

    # Summarize
    def summarize(vals):
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        se = s / np.sqrt(len(vals))
        lower = m - 1.96 * se
        upper = m + 1.96 * se
        return {"mean": m, "std": s, "95%_CI": (lower, upper)}

    return {
        "RMSE": summarize(rmse_list),
        "MAE": summarize(mae_list),
        "raw_rmse": rmse_list,
        "raw_mae": mae_list
    }


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
    
    def estimate_uncertainty(
        self,
        X: pd.DataFrame,
        n_repeats: int = 100,
        mask_strategy: str = 'block',
        frac: float = 0.1,
        block_len: int = 5,
        n_blocks: int = 1,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate imputation uncertainty using Monte Carlo validation.
        
        This method temporarily masks observed values, imputes them, and
        compares with actual values to estimate prediction error.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
        n_repeats : int, optional
            Number of Monte Carlo iterations (default: 100)
        mask_strategy : str, optional
            'random' or 'block' masking strategy (default: 'block')
        frac : float, optional
            Fraction of values to mask for 'random' strategy (default: 0.1)
        block_len : int, optional
            Length of temporal blocks for 'block' strategy (default: 5)
        n_blocks : int, optional
            Number of blocks for 'block' strategy (default: 1)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'RMSE': {'mean': float, 'std': float, '95%_CI': tuple}
            - 'MAE': {'mean': float, 'std': float, '95%_CI': tuple}
            - 'raw_rmse': list of individual RMSE values
            - 'raw_mae': list of individual MAE values
            
        Examples
        --------
        >>> imputer = Imputer()
        >>> imputer.fit(df)
        >>> uncertainty = imputer.estimate_uncertainty(df, n_repeats=100)
        >>> print(f"RMSE: {uncertainty['RMSE']['mean']:.3f}")
        """
        # Check if fitted
        if not self.is_fitted_:
            raise RuntimeError(
                "Imputer must be fitted before estimating uncertainty. "
                "Call fit() first."
            )
        
        # Validate input data
        X_validated = validate_dataframe(X)
        
        # Convert to numpy array
        X_array = X_validated.values
        
        if self.verbose:
            print(f"Estimating uncertainty with {n_repeats} Monte Carlo repeats...")
            print(f"Strategy: {mask_strategy}, Rank: {self.rank_}")
        
        # Perform Monte Carlo validation
        results = _monte_carlo_validation(
            X_array,
            rank=self.rank_,
            scaler=self.scaler,
            max_iters=self.max_iters,
            tol=self.tol,
            n_repeats=n_repeats,
            mask_strategy=mask_strategy,
            frac=frac,
            block_len=block_len,
            n_blocks=n_blocks,
            seed=seed
        )
        
        if self.verbose:
            print(f"RMSE: {results['RMSE']['mean']:.4f} "
                  f"(95% CI: {results['RMSE']['95%_CI'][0]:.4f} - "
                  f"{results['RMSE']['95%_CI'][1]:.4f})")
            print(f"MAE:  {results['MAE']['mean']:.4f} "
                  f"(95% CI: {results['MAE']['95%_CI'][0]:.4f} - "
                  f"{results['MAE']['95%_CI'][1]:.4f})")
        
        return results
    
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
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = False,
        uncertainty_method: str = 'monte_carlo',
        n_repeats: int = 100,
        n_bootstrap: int = 50,
        mask_strategy: str = 'block',
        frac: float = 0.1,
        block_len: int = 5,
        n_blocks: int = 1,
        confidence: float = 0.95,
        seed: Optional[int] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Fit the imputer and transform the data in one step.
        
        Optionally compute uncertainty estimates using one of three methods:
        - 'monte_carlo': Constant uncertainty band from validation (fast)
        - 'bootstrap': Point-wise uncertainty via resampling (slower, more accurate)
        - 'hybrid': Combines both approaches
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
        return_uncertainty : bool, optional
            Whether to return uncertainty estimates (default: False)
        uncertainty_method : str, optional
            Method for uncertainty estimation: 'monte_carlo', 'bootstrap', or 'hybrid'
            (default: 'monte_carlo')
        n_repeats : int, optional
            Number of Monte Carlo repeats for validation (default: 100)
        n_bootstrap : int, optional
            Number of bootstrap samples for 'bootstrap' and 'hybrid' methods (default: 50)
        mask_strategy : str, optional
            Masking strategy for Monte Carlo: 'random' or 'block' (default: 'block')
        frac : float, optional
            Fraction to mask for 'random' strategy (default: 0.1)
        block_len : int, optional
            Block length for 'block' strategy (default: 5)
        n_blocks : int, optional
            Number of blocks for 'block' strategy (default: 1)
        confidence : float, optional
            Confidence level for intervals (default: 0.95)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        pd.DataFrame or tuple
            If return_uncertainty=False: Returns imputed DataFrame
            If return_uncertainty=True: Returns (imputed_df, uncertainty_dict)
            
        Examples
        --------
        >>> # Simple imputation (no uncertainty)
        >>> imputer = Imputer()
        >>> df_imputed = imputer.fit_transform(df)
        
        >>> # With Monte Carlo uncertainty (constant band)
        >>> df_imputed, unc = imputer.fit_transform(
        ...     df, 
        ...     return_uncertainty=True,
        ...     uncertainty_method='monte_carlo'
        ... )
        >>> print(f"RMSE: {unc['rmse']:.3f}")
        
        >>> # With bootstrap uncertainty (point-wise)
        >>> df_imputed, unc = imputer.fit_transform(
        ...     df,
        ...     return_uncertainty=True,
        ...     uncertainty_method='bootstrap',
        ...     n_bootstrap=50
        ... )
        >>> df_lower, df_upper = unc['lower'], unc['upper']
        """
        if not return_uncertainty:
            # Standard behavior - no uncertainty
            return self.fit(X).transform(X)
        
        # Validate uncertainty method
        valid_methods = ['monte_carlo', 'bootstrap', 'hybrid']
        if uncertainty_method not in valid_methods:
            raise ValueError(
                f"uncertainty_method must be one of {valid_methods}, "
                f"got '{uncertainty_method}'"
            )
        
        # Fit and transform
        self.fit(X)
        df_imputed = self.transform(X)
        
        # Compute uncertainty based on method
        if uncertainty_method == 'monte_carlo':
            uncertainty_dict = self._uncertainty_monte_carlo(
                X, n_repeats, mask_strategy, frac, block_len, n_blocks, seed
            )
        elif uncertainty_method == 'bootstrap':
            uncertainty_dict = self._uncertainty_bootstrap(
                X, df_imputed, n_bootstrap, confidence, seed
            )
        elif uncertainty_method == 'hybrid':
            uncertainty_dict = self._uncertainty_hybrid(
                X, df_imputed, n_repeats, n_bootstrap, mask_strategy,
                frac, block_len, n_blocks, confidence, seed
            )
        
        return df_imputed, uncertainty_dict
    
    def _uncertainty_monte_carlo(
        self,
        X: pd.DataFrame,
        n_repeats: int,
        mask_strategy: str,
        frac: float,
        block_len: int,
        n_blocks: int,
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """
        Compute constant uncertainty band using Monte Carlo validation.
        
        Returns
        -------
        dict
            {'method': 'monte_carlo', 'rmse': float, 'mae': float, 
             'rmse_ci': tuple, 'mae_ci': tuple}
        """
        results = self.estimate_uncertainty(
            X, n_repeats, mask_strategy, frac, block_len, n_blocks, seed
        )
        
        return {
            'method': 'monte_carlo',
            'rmse': results['RMSE']['mean'],
            'mae': results['MAE']['mean'],
            'rmse_std': results['RMSE']['std'],
            'mae_std': results['MAE']['std'],
            'rmse_ci': results['RMSE']['95%_CI'],
            'mae_ci': results['MAE']['95%_CI'],
            'raw_rmse': results['raw_rmse'],
            'raw_mae': results['raw_mae']
        }
    
    def _uncertainty_bootstrap(
        self,
        X: pd.DataFrame,
        df_imputed: pd.DataFrame,
        n_bootstrap: int,
        confidence: float,
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """
        Compute point-wise uncertainty using bootstrap resampling.
        
        Returns
        -------
        dict
            {'method': 'bootstrap', 'lower': DataFrame, 'upper': DataFrame,
             'std': DataFrame}
        """
        if self.verbose:
            print(f"Computing bootstrap uncertainty with {n_bootstrap} samples...")
        
        rng = np.random.default_rng(seed)
        X_validated = validate_dataframe(X)
        X_array = X_validated.values
        
        # Store bootstrap predictions for each missing value
        missing_mask = np.isnan(X_array)
        n_missing = missing_mask.sum()
        
        if n_missing == 0:
            warnings.warn("No missing values to compute uncertainty for")
            return {
                'method': 'bootstrap',
                'lower': df_imputed.copy(),
                'upper': df_imputed.copy(),
                'std': pd.DataFrame(0.0, index=df_imputed.index, columns=df_imputed.columns)
            }
        
        # Collect bootstrap samples
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            # Different random seed for each bootstrap
            s = None if seed is None else seed + i
            
            # Create a modified version by bootstrapping observed values
            X_boot = X_array.copy()
            
            # For each column, bootstrap the observed values
            for col_idx in range(X_boot.shape[1]):
                col_data = X_boot[:, col_idx]
                observed_mask = ~np.isnan(col_data)
                observed_values = col_data[observed_mask]
                
                if len(observed_values) > 0:
                    # Bootstrap resample observed values
                    rng_boot = np.random.default_rng(s + col_idx if s else None)
                    resampled_values = rng_boot.choice(
                        observed_values,
                        size=len(observed_values),
                        replace=True
                    )
                    X_boot[observed_mask, col_idx] = resampled_values
            
            # Perform imputation on bootstrapped data
            X_imputed_boot = iterative_svd_impute(
                X_boot,
                rank=self.rank_,
                max_iters=self.max_iters,
                tol=self.tol,
                scaler=self.scaler
            )
            
            bootstrap_predictions.append(X_imputed_boot)
        
        # Stack predictions
        bootstrap_stack = np.stack(bootstrap_predictions, axis=0)
        
        # Compute statistics
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        mean_pred = np.mean(bootstrap_stack, axis=0)
        std_pred = np.std(bootstrap_stack, axis=0)
        lower_pred = np.percentile(bootstrap_stack, lower_percentile, axis=0)
        upper_pred = np.percentile(bootstrap_stack, upper_percentile, axis=0)
        
        # Convert to DataFrames
        df_lower = pd.DataFrame(lower_pred, index=X_validated.index, columns=X_validated.columns)
        df_upper = pd.DataFrame(upper_pred, index=X_validated.index, columns=X_validated.columns)
        df_std = pd.DataFrame(std_pred, index=X_validated.index, columns=X_validated.columns)
        
        if self.verbose:
            avg_std = std_pred[missing_mask].mean()
            print(f"Average uncertainty (std): {avg_std:.4f}")
        
        return {
            'method': 'bootstrap',
            'lower': df_lower,
            'upper': df_upper,
            'std': df_std,
            'confidence': confidence
        }
    
    def _uncertainty_hybrid(
        self,
        X: pd.DataFrame,
        df_imputed: pd.DataFrame,
        n_repeats: int,
        n_bootstrap: int,
        mask_strategy: str,
        frac: float,
        block_len: int,
        n_blocks: int,
        confidence: float,
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """
        Compute uncertainty using hybrid approach (Monte Carlo + Bootstrap).
        
        Returns
        -------
        dict
            Combination of both methods
        """
        if self.verbose:
            print("Computing hybrid uncertainty...")
        
        # Get Monte Carlo results
        mc_results = self._uncertainty_monte_carlo(
            X, n_repeats, mask_strategy, frac, block_len, n_blocks, seed
        )
        
        # Get bootstrap results
        bootstrap_results = self._uncertainty_bootstrap(
            X, df_imputed, n_bootstrap, confidence, seed
        )
        
        # Combine: use bootstrap intervals scaled by MC validation
        # Calculate scale factor using available data
        X_validated = validate_dataframe(X)
        missing_mask = np.isnan(X_validated.values)
        
        if missing_mask.any():
            # Use std from bootstrap for missing values only
            bootstrap_std_missing = bootstrap_results['std'].values[missing_mask]
            avg_bootstrap_std = bootstrap_std_missing[~np.isnan(bootstrap_std_missing)].mean()
            
            if avg_bootstrap_std > 0:
                scale_factor = mc_results['rmse'] / avg_bootstrap_std
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0
        
        df_lower_scaled = df_imputed - bootstrap_results['std'] * scale_factor * 1.96
        df_upper_scaled = df_imputed + bootstrap_results['std'] * scale_factor * 1.96
        
        return {
            'method': 'hybrid',
            'lower': df_lower_scaled,
            'upper': df_upper_scaled,
            'std': bootstrap_results['std'] * scale_factor,
            'monte_carlo': mc_results,
            'bootstrap': bootstrap_results,
            'confidence': confidence
        }
    
    def get_confidence_intervals(
        self,
        df_imputed: pd.DataFrame,
        uncertainty_dict: Dict[str, Any],
        confidence: float = 0.95
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract confidence intervals from uncertainty dictionary.
        
        Parameters
        ----------
        df_imputed : pd.DataFrame
            Imputed DataFrame
        uncertainty_dict : dict
            Uncertainty dictionary from fit_transform
        confidence : float, optional
            Confidence level (only used for 'monte_carlo' method)
            
        Returns
        -------
        tuple of pd.DataFrame
            (lower_bound, upper_bound) DataFrames
            
        Examples
        --------
        >>> df_imputed, unc = imputer.fit_transform(df, return_uncertainty=True)
        >>> df_lower, df_upper = imputer.get_confidence_intervals(df_imputed, unc)
        """
        method = uncertainty_dict['method']
        
        if method == 'monte_carlo':
            # Constant band based on RMSE
            rmse = uncertainty_dict['rmse']
            z_score = 1.96 if confidence == 0.95 else 2.576  # Simplified
            band = rmse * z_score
            
            df_lower = df_imputed - band
            df_upper = df_imputed + band
            
        elif method in ['bootstrap', 'hybrid']:
            # Use pre-computed intervals
            df_lower = uncertainty_dict['lower']
            df_upper = uncertainty_dict['upper']
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df_lower, df_upper
    
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
