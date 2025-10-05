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
    rank : int, str, or None, optional
        Fixed rank to use. Options:
        - int: Use fixed rank value
        - "auto": Optimize rank via cross-validation to minimize imputation error
        - None (default): Estimate rank based on variance_threshold
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
    optimization_results_ : dict, optional
        Results from rank optimization (only set when rank="auto")
        
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
    >>> 
    >>> # Auto-optimize rank via cross-validation
    >>> imputer = Imputer(rank="auto")
    >>> df_imputed = imputer.fit_transform(df)
    >>> print(f"Optimized rank: {imputer.rank_}")
    >>> 
    >>> # Auto-optimize rank via cross-validation
    >>> imputer = Imputer(rank="auto")
    >>> df_imputed = imputer.fit_transform(df)
    >>> print(f"Optimized rank: {imputer.rank_}")
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.95,
        rank: Union[int, str, None] = None,
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
        
        if rank is not None:
            if isinstance(rank, str):
                if rank != "auto":
                    raise ValueError(f"Only 'auto' is supported as string for rank, got '{rank}'")
            elif not isinstance(rank, int) or rank < 1:
                raise ValueError(f"rank must be a positive integer or 'auto', got {rank}")
        
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
        self.optimization_results_ = None
    
    def fit(self, X: pd.DataFrame) -> 'Imputer':
        """
        Fit the imputer on the data (estimate or optimize rank if needed).
        
        The rank determination strategy depends on the `rank` parameter:
        - If rank="auto": Optimize rank via cross-validation to minimize error
        - If rank=None: Estimate rank based on variance_threshold (default)
        - If rank=int: Use the specified fixed rank
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
            
        Returns
        -------
        self
            Fitted imputer
            
        Notes
        -----
        When rank="auto", the optimization results are stored in the
        `optimization_results_` attribute for inspection.
        """
        # Validate input data
        X_validated = validate_dataframe(X)
        
        # Store metadata
        self.columns_ = X_validated.columns.tolist()
        self.index_name_ = X_validated.index.name
        
        # Convert to numpy array
        X_array = X_validated.values
        
        # Determine rank based on user specification
        if self.rank == "auto":
            # Auto-optimize rank via cross-validation
            if self.verbose:
                print("Auto-optimizing rank via cross-validation...")
            
            self.optimization_results_ = self.optimize_rank(X_validated)
            self.rank_ = self.optimization_results_['optimal_rank']
            
            if self.verbose:
                score = self.optimization_results_['optimal_score']
                converged = self.optimization_results_['convergence_info']['is_converged']
                print(f"Optimized rank: {self.rank_} (CV score: {score:.4f})")
                if not converged:
                    print("Warning: Optimization may not have converged to a clear minimum")
        
        elif self.rank is None:
            # Variance-based estimation (default behavior)
            self.rank_ = estimate_rank(X_array, self.variance_threshold)
            if self.verbose:
                print(f"Estimated rank: {self.rank_} "
                      f"(variance threshold: {self.variance_threshold*100:.0f}%)")
        
        else:
            # Fixed rank specified by user
            self.rank_ = self.rank
            # Check if requested rank is feasible
            check_sufficient_rank(X_validated, self.rank_)
            if self.verbose:
                print(f"Using fixed rank: {self.rank_}")
        
        self.is_fitted_ = True
        return self
    
    def optimize_rank(
        self,
        X: pd.DataFrame,
        rank_range: Optional[Tuple[int, int]] = None,
        cv_folds: int = 5,
        n_repeats_per_fold: int = 20,
        mask_strategy: str = 'random',
        frac: float = 0.1,
        block_len: int = 5,
        n_blocks: int = 1,
        metric: str = 'rmse',
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize rank via cross-validation to minimize imputation error.
        
        This method systematically tests different rank values using cross-validation
        with multiple random masking experiments to find the rank that minimizes
        prediction error on held-out data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with datetime index
        rank_range : tuple of int, optional
            (min_rank, max_rank) to test. If None, uses [1, min(n_rows, n_cols, 10)]
        cv_folds : int, optional
            Number of cross-validation folds (default: 5)
        n_repeats_per_fold : int, optional
            Number of random masking experiments per fold (default: 20)
        mask_strategy : str, optional
            'random' or 'block' masking strategy (default: 'random')
        frac : float, optional
            Fraction of values to mask for 'random' strategy (default: 0.1)
        block_len : int, optional
            Block length for 'block' strategy (default: 5)
        n_blocks : int, optional
            Number of blocks for 'block' strategy (default: 1)
        metric : str, optional
            Optimization metric: 'rmse' or 'mae' (default: 'rmse')
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'optimal_rank': Best rank found
            - 'optimal_score': Cross-validation score for optimal rank
            - 'results_df': DataFrame with detailed results for all ranks
            - 'cv_details': Fold-by-fold results
            - 'convergence_info': Optimization diagnostics
            
        Examples
        --------
        >>> imputer = Imputer()
        >>> results = imputer.optimize_rank(df, rank_range=(1, 8))
        >>> print(f"Optimal rank: {results['optimal_rank']}")
        >>> print(results['results_df'])
        """
        # Validate input data
        X_validated = validate_dataframe(X)
        X_array = X_validated.values
        n_rows, n_cols = X_array.shape
        
        # Determine rank range
        if rank_range is None:
            max_possible = min(n_rows, n_cols)
            max_test = min(max_possible, 10)  # Reasonable upper bound
            rank_range = (1, max_test)
        
        min_rank, max_rank = rank_range
        if min_rank < 1:
            raise ValueError(f"min_rank must be at least 1, got {min_rank}")
        if max_rank > min(n_rows, n_cols):
            raise ValueError(
                f"max_rank={max_rank} exceeds maximum possible rank {min(n_rows, n_cols)} "
                f"for matrix with shape {X_array.shape}"
            )
        
        if metric not in ['rmse', 'mae']:
            raise ValueError(f"metric must be 'rmse' or 'mae', got '{metric}'")
        
        if self.verbose:
            print(f"Optimizing rank in range [{min_rank}, {max_rank}] using {cv_folds}-fold CV")
            print(f"Strategy: {mask_strategy}, Repeats per fold: {n_repeats_per_fold}")
        
        # Initialize random number generator
        rng = np.random.default_rng(seed)
        
        # Storage for results
        rank_results = []
        cv_details = {}
        
        # Test each rank
        ranks_to_test = list(range(min_rank, max_rank + 1))
        
        for rank in ranks_to_test:
            if self.verbose:
                print(f"Testing rank {rank}...")
            
            fold_scores = []
            cv_details[rank] = []
            
            # Cross-validation folds
            for fold in range(cv_folds):
                fold_seed = None if seed is None else int(rng.integers(1 << 30))
                
                # Perform validation for this fold and rank
                fold_results = _monte_carlo_validation(
                    X_array,
                    rank=rank,
                    scaler=self.scaler,
                    max_iters=self.max_iters,
                    tol=self.tol,
                    n_repeats=n_repeats_per_fold,
                    mask_strategy=mask_strategy,
                    frac=frac,
                    block_len=block_len,
                    n_blocks=n_blocks,
                    seed=fold_seed
                )
                
                # Extract the metric of interest
                if metric == 'rmse':
                    fold_score = fold_results['RMSE']['mean']
                else:  # mae
                    fold_score = fold_results['MAE']['mean']
                
                fold_scores.append(fold_score)
                cv_details[rank].append({
                    'fold': fold,
                    'score': fold_score,
                    'full_results': fold_results
                })
            
            # Summarize across folds
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            rank_results.append({
                'rank': rank,
                f'mean_{metric}': mean_score,
                f'std_{metric}': std_score,
                'fold_scores': fold_scores
            })
            
            if self.verbose:
                print(f"  Rank {rank}: {metric.upper()}={mean_score:.4f} ± {std_score:.4f}")
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(rank_results)
        
        # Find optimal rank (minimum error)
        optimal_idx = results_df[f'mean_{metric}'].idxmin()
        optimal_rank = results_df.loc[optimal_idx, 'rank']
        optimal_score = results_df.loc[optimal_idx, f'mean_{metric}']
        
        # Check convergence (is there a clear minimum?)
        scores = results_df[f'mean_{metric}'].values
        stds = results_df[f'std_{metric}'].values
        
        # Simple convergence check: optimal score should be significantly better than others
        other_scores = scores[scores != optimal_score]
        if len(other_scores) > 0:
            min_other = other_scores.min()
            improvement = min_other - optimal_score
            significance = improvement / stds[optimal_idx] if stds[optimal_idx] > 0 else float('inf')
        else:
            significance = float('inf')
        
        convergence_info = {
            'improvement_over_second_best': improvement if len(other_scores) > 0 else 0,
            'significance_ratio': significance,
            'is_converged': significance > 1.0,  # At least 1 std dev improvement
            'tested_ranks': ranks_to_test,
            'total_experiments': len(ranks_to_test) * cv_folds * n_repeats_per_fold
        }
        
        if self.verbose:
            print(f"\nOptimization complete:")
            print(f"  Optimal rank: {optimal_rank}")
            print(f"  {metric.upper()}: {optimal_score:.4f}")
            print(f"  Convergence: {'Yes' if convergence_info['is_converged'] else 'No'}")
            if not convergence_info['is_converged']:
                print(f"  Consider expanding rank range or increasing n_repeats_per_fold")
        
        return {
            'optimal_rank': int(optimal_rank),
            'optimal_score': float(optimal_score),
            'results_df': results_df,
            'cv_details': cv_details,
            'convergence_info': convergence_info,
            'parameters': {
                'rank_range': rank_range,
                'cv_folds': cv_folds,
                'n_repeats_per_fold': n_repeats_per_fold,
                'mask_strategy': mask_strategy,
                'metric': metric,
                'seed': seed
            }
        }
    
    def get_optimization_results(self) -> Optional[Dict[str, Any]]:
        """
        Get rank optimization results (only available when rank="auto" was used).
        
        Returns
        -------
        dict or None
            Optimization results dictionary containing:
            - 'optimal_rank': Best rank found
            - 'optimal_score': Cross-validation score
            - 'results_df': DataFrame with all tested ranks
            - 'cv_details': Detailed fold-by-fold results
            - 'convergence_info': Optimization diagnostics
            - 'parameters': Optimization parameters used
            
        Examples
        --------
        >>> imputer = Imputer(rank="auto")
        >>> imputer.fit(df)
        >>> results = imputer.get_optimization_results()
        >>> print(results['results_df'])
        >>> print(f"Tested {len(results['cv_details'])} ranks")
        """
        if not hasattr(self, 'optimization_results_') or self.optimization_results_ is None:
            return None
        return self.optimization_results_
    
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
    
    def _learn_distance_uncertainty_relationship(
        self, 
        df: pd.DataFrame, 
        n_samples: int = 50
    ):
        """
        Learn empirical relationship between distance to nearest observation
        and prediction error through targeted validation sampling.
        
        This method:
        1. Masks random observed values one at a time
        2. Imputes and measures error
        3. Records distance to nearest remaining observation
        4. Fits a model relating distance to error
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data with missing values
        n_samples : int
            Number of validation samples to collect
            
        Returns
        -------
        callable
            Function that maps distance (days) to uncertainty multiplier
        """
        distances = []
        errors = []
        
        
        for _ in range(n_samples):
            # Randomly mask one observation per column
            masked_df = df.copy()
            mask_locations = {}
            
            for col in df.columns:
                observed_idx = df[col].dropna().index
                if len(observed_idx) > 2:
                    # Mask a random observation
                    mask_idx = np.random.choice(observed_idx)
                    mask_locations[col] = (mask_idx, masked_df.loc[mask_idx, col])
                    masked_df.loc[mask_idx, col] = np.nan
            
            if not mask_locations:
                continue
            
            # Impute
            X_masked = masked_df.values
            try:
                X_imputed = iterative_svd_impute(
                    X_masked,
                    rank=self.rank_,
                    max_iters=self.max_iters,
                    tol=self.tol,
                    scaler=self.scaler
                )
                imputed_df = pd.DataFrame(
                    X_imputed, 
                    index=masked_df.index, 
                    columns=masked_df.columns
                )
            except:
                continue
            
            # Calculate error and distance for each masked location
            for col, (mask_idx, true_val) in mask_locations.items():
                pred_val = imputed_df.loc[mask_idx, col]
                error = abs(pred_val - true_val)
                
                # Distance to nearest observation (in days)
                remaining_obs = masked_df[col].dropna().index
                if len(remaining_obs) > 0:
                    time_diffs = abs((remaining_obs - mask_idx).total_seconds() / 86400)
                    nearest_dist = time_diffs.min()
                    
                    distances.append(nearest_dist)
                    errors.append(error)
        
        if len(distances) == 0:
            # No valid distances found, return identity function
            if self.verbose:
                print("Warning: Could not learn distance-uncertainty relationship, using identity")
            return lambda d: 1.0
        
        distances = np.array(distances)
        errors = np.array(errors)
        
        # Remove outliers
        valid_mask = (errors < np.percentile(errors, 95))
        distances = distances[valid_mask]
        errors = errors[valid_mask]
        
        if len(distances) < 10:
            # Not enough data, return identity function
            if self.verbose:
                print(f"Warning: Only {len(distances)} samples for distance learning, using identity")
            return lambda d: 1.0
        
        # Bin by distance and compute mean error
        try:
            from scipy.stats import binned_statistic
            n_bins = min(5, max(3, len(distances) // 10))
            bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
            bin_means, _, _ = binned_statistic(
                distances, errors, statistic='mean', bins=bin_edges
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Fit exponential growth model: error ~ a * exp(b * distance)
            from scipy.optimize import curve_fit
            
            def exp_model(d, a, b):
                return a * np.exp(b * d)
            
            # Try exponential fit
            try:
                popt, _ = curve_fit(
                    exp_model, bin_centers, bin_means, 
                    p0=[bin_means[0], 0.01], maxfev=1000
                )
                baseline = exp_model(0, *popt)
                if baseline > 0 and popt[1] >= 0:  # Ensure positive growth
                    return lambda d: exp_model(d, *popt) / baseline
            except:
                pass
            
            # Fallback to linear fit
            if len(distances) > 1:
                slope, intercept = np.polyfit(distances, errors, 1)
                if intercept > 0 and slope >= 0:  # Ensure positive
                    return lambda d: (intercept + slope * d) / intercept
        except:
            pass
        
        # Final fallback: use median error ratio
        median_dist = np.median(distances)
        near_errors = errors[distances <= median_dist]
        far_errors = errors[distances > median_dist]
        
        if len(near_errors) > 0 and len(far_errors) > 0:
            ratio = max(1.0, np.median(far_errors) / np.median(near_errors))
            return lambda d: 1.0 if d <= median_dist else ratio
        
        return lambda d: 1.0
    
    def _apply_proximity_adjustment(
        self, 
        df: pd.DataFrame, 
        uncertainty_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adjust uncertainty estimates based on learned distance-error relationship.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original data with missing values
        uncertainty_df : pd.DataFrame
            Initial uncertainty estimates (std or CI width)
            
        Returns
        -------
        pd.DataFrame
            Adjusted uncertainty estimates
        """
        if self.verbose:
            print("Learning distance-uncertainty relationship from data...")
        
        # Learn the relationship
        distance_fn = self._learn_distance_uncertainty_relationship(df, n_samples=50)
        
        # Store for inspection
        self.distance_to_uncertainty_fn_ = distance_fn
        
        adjusted_unc = uncertainty_df.copy()
        
        for col in df.columns:
            missing_mask = df[col].isna()
            if not missing_mask.any():
                continue
                
            observed_times = df[col].dropna().index
            
            if len(observed_times) == 0:
                continue
            
            for idx in df.index[missing_mask]:
                # Calculate distance to nearest observation (days)
                time_diffs = abs((observed_times - idx).total_seconds() / 86400)
                nearest_dist = time_diffs.min()
                
                # Apply learned adjustment
                multiplier = distance_fn(nearest_dist)
                adjusted_unc.loc[idx, col] *= multiplier
        
        if self.verbose:
            print(f"Applied proximity-based uncertainty adjustment")
            # Show some example multipliers
            test_distances = [0, 7, 30, 90, 365]
            print("Distance -> Uncertainty multiplier:")
            for d in test_distances:
                print(f"  {d:4d} days: {distance_fn(d):.3f}x")
        
        return adjusted_unc
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = False,
        uncertainty_method: str = 'monte_carlo',
        adjust_by_proximity: bool = False,
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
        adjust_by_proximity : bool, optional
            If True, adjust uncertainty based on learned distance-error relationship.
            This is fully data-driven and requires no user parameters. (default: False)
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
        
        >>> # With proximity-adjusted uncertainty (data-driven)
        >>> df_imputed, unc = imputer.fit_transform(
        ...     df,
        ...     return_uncertainty=True,
        ...     uncertainty_method='hybrid',
        ...     adjust_by_proximity=True
        ... )
        >>> # Uncertainty now varies based on distance to observations
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
        
        # Apply proximity-based adjustment if requested
        if adjust_by_proximity:
            if 'std' in uncertainty_dict:
                # Adjust standard deviation
                uncertainty_dict['std'] = self._apply_proximity_adjustment(
                    X, uncertainty_dict['std']
                )
            if 'lower' in uncertainty_dict and 'upper' in uncertainty_dict:
                # Adjust confidence intervals
                ci_width = (uncertainty_dict['upper'] - uncertainty_dict['lower']) / 2
                adjusted_width = self._apply_proximity_adjustment(X, ci_width)
                uncertainty_dict['lower'] = df_imputed - adjusted_width
                uncertainty_dict['upper'] = df_imputed + adjusted_width
        
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
    
    def condition_on_observations(
        self,
        df_original: pd.DataFrame,
        df_imputed: pd.DataFrame,
        uncertainty_dict: Optional[Dict[str, Any]] = None,
        temporal_range: float = 30.0,
        spatial_weight: float = 0.5
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Condition imputed values on known observations using Kriging.
        
        This method refines imputed values and reduces uncertainty by:
        1. Temporal conditioning: Force imputed gaps to match boundary observations
        2. Spatial conditioning: Use correlations with other series
        3. Uncertainty reduction: Apply Kriging variance formula
        
        Parameters
        ----------
        df_original : pd.DataFrame
            Original data with missing values (NaN)
        df_imputed : pd.DataFrame
            Imputed data (no NaN)
        uncertainty_dict : dict, optional
            Uncertainty estimates from fit_transform. If provided, will return
            conditioned uncertainty.
        temporal_range : float, optional
            Characteristic time scale (days) for temporal correlation decay.
            Default: 30 days
        spatial_weight : float, optional
            Weight for spatial (cross-series) conditioning [0, 1].
            Default: 0.5 (equal weight to temporal and spatial)
            
        Returns
        -------
        pd.DataFrame or tuple
            If uncertainty_dict is None: Returns conditioned DataFrame
            If uncertainty_dict provided: Returns (conditioned_df, conditioned_uncertainty_dict)
            
        Examples
        --------
        >>> # Basic conditioning (no uncertainty)
        >>> df_conditioned = imputer.condition_on_observations(df_original, df_imputed)
        
        >>> # With uncertainty reduction
        >>> df_imputed, unc = imputer.fit_transform(df, return_uncertainty=True)
        >>> df_cond, unc_cond = imputer.condition_on_observations(
        ...     df, df_imputed, unc
        ... )
        """
        df_conditioned = df_imputed.copy()
        
        # Compute cross-series correlation for spatial conditioning
        corr_matrix = df_original.corr()
        
        if self.verbose:
            print("Conditioning imputed values on observations...")
            print(f"  Temporal range: {temporal_range} days")
            print(f"  Spatial weight: {spatial_weight}")
        
        # Process each column
        for col in df_original.columns:
            missing_mask = df_original[col].isna()
            observed_times = df_original[col].dropna().index
            
            if not missing_mask.any() or len(observed_times) < 2:
                continue
            
            # Get other correlated series for spatial conditioning
            other_cols = [c for c in df_original.columns if c != col]
            correlations = corr_matrix.loc[col, other_cols].abs()
            
            # Process each missing value
            for miss_time in df_original.index[missing_mask]:
                # === TEMPORAL CONDITIONING ===
                # Find nearest observations before and after
                times_before = observed_times[observed_times < miss_time]
                times_after = observed_times[observed_times > miss_time]
                
                temporal_correction = 0.0
                temporal_weight_sum = 0.0
                
                # Kriging weights based on exponential decay
                for obs_time in observed_times:
                    dt_days = abs((miss_time - obs_time).total_seconds() / 86400)
                    
                    # Exponential correlation model
                    correlation = np.exp(-dt_days / temporal_range)
                    
                    if correlation > 0.01:  # Only use significant correlations
                        # Residual at observation point
                        residual = df_original.loc[obs_time, col] - df_imputed.loc[obs_time, col]
                        
                        temporal_correction += correlation * residual
                        temporal_weight_sum += correlation
                
                if temporal_weight_sum > 0:
                    temporal_correction /= temporal_weight_sum
                
                # === SPATIAL CONDITIONING ===
                spatial_correction = 0.0
                spatial_weight_sum = 0.0
                
                for other_col in other_cols:
                    if pd.notna(df_original.loc[miss_time, other_col]):
                        # Other series has observation at this time
                        corr_value = correlations[other_col]
                        
                        if corr_value > 0.3:  # Only use moderately correlated series
                            # Check if imputation agrees with cross-series relationship
                            # Use regression: col ~ other_col
                            obs_mask = df_original[col].notna() & df_original[other_col].notna()
                            if obs_mask.sum() > 5:
                                # Simple linear relationship
                                x = df_original.loc[obs_mask, other_col].values
                                y = df_original.loc[obs_mask, col].values
                                
                                # Fit: y = a*x + b
                                a, b = np.polyfit(x, y, 1)
                                
                                # Predicted value based on other series
                                predicted = a * df_original.loc[miss_time, other_col] + b
                                residual = predicted - df_imputed.loc[miss_time, col]
                                
                                spatial_correction += corr_value * residual
                                spatial_weight_sum += corr_value
                
                if spatial_weight_sum > 0:
                    spatial_correction /= spatial_weight_sum
                
                # === COMBINE CORRECTIONS ===
                total_correction = (
                    (1 - spatial_weight) * temporal_correction +
                    spatial_weight * spatial_correction
                )
                
                df_conditioned.loc[miss_time, col] = df_imputed.loc[miss_time, col] + total_correction
        
        if self.verbose:
            avg_change = (df_conditioned - df_imputed).abs().mean().mean()
            print(f"  Average adjustment: {avg_change:.4f}")
        
        # If uncertainty provided, reduce it based on conditioning
        if uncertainty_dict is not None:
            conditioned_uncertainty = self._reduce_uncertainty_kriging(
                df_original, df_imputed, df_conditioned,
                uncertainty_dict, temporal_range, spatial_weight
            )
            return df_conditioned, conditioned_uncertainty
        
        return df_conditioned
    
    def _reduce_uncertainty_kriging(
        self,
        df_original: pd.DataFrame,
        df_imputed: pd.DataFrame,
        df_conditioned: pd.DataFrame,
        uncertainty_dict: Dict[str, Any],
        temporal_range: float,
        spatial_weight: float
    ) -> Dict[str, Any]:
        """
        Reduce uncertainty based on Kriging variance formula.
        
        Kriging variance: σ²_conditioned = σ²_prior * (1 - ρ²)
        where ρ² is the squared correlation due to conditioning.
        """
        conditioned_unc = uncertainty_dict.copy()
        method = uncertainty_dict['method']
        
        # Compute correlation matrix
        corr_matrix = df_original.corr()
        
        if method == 'monte_carlo':
            # Adjust RMSE
            avg_reduction = self._compute_average_kriging_reduction(
                df_original, temporal_range, spatial_weight, corr_matrix
            )
            conditioned_unc['rmse'] *= np.sqrt(1 - avg_reduction)
            conditioned_unc['mae'] *= np.sqrt(1 - avg_reduction)
            
        elif method in ['bootstrap', 'hybrid']:
            # Adjust point-wise uncertainty
            if 'std' in conditioned_unc:
                std_df = conditioned_unc['std'].copy()
                
                for col in df_original.columns:
                    missing_mask = df_original[col].isna()
                    
                    for miss_time in df_original.index[missing_mask]:
                        # Compute ρ² for this specific point
                        rho_squared = self._compute_kriging_correlation(
                            df_original, col, miss_time,
                            temporal_range, spatial_weight, corr_matrix
                        )
                        
                        # Reduce uncertainty
                        std_df.loc[miss_time, col] *= np.sqrt(1 - rho_squared)
                
                conditioned_unc['std'] = std_df
            
            # Adjust confidence intervals
            if 'lower' in conditioned_unc and 'upper' in conditioned_unc:
                # Recompute intervals with reduced std
                ci_half_width = (conditioned_unc['upper'] - df_conditioned) * np.sqrt(
                    1 - self._compute_average_kriging_reduction(
                        df_original, temporal_range, spatial_weight, corr_matrix
                    )
                )
                conditioned_unc['lower'] = df_conditioned - ci_half_width
                conditioned_unc['upper'] = df_conditioned + ci_half_width
        
        if self.verbose:
            print(f"  Uncertainty reduced by Kriging conditioning")
        
        return conditioned_unc
    
    def _compute_kriging_correlation(
        self,
        df_original: pd.DataFrame,
        col: str,
        miss_time: pd.Timestamp,
        temporal_range: float,
        spatial_weight: float,
        corr_matrix: pd.DataFrame
    ) -> float:
        """
        Compute ρ² (squared correlation) for a specific missing point.
        
        Returns value in [0, 1] representing conditioning strength.
        """
        # Temporal correlation
        observed_times = df_original[col].dropna().index
        
        if len(observed_times) == 0:
            return 0.0
        
        # Find closest observations
        dt_days = np.array([abs((miss_time - t).total_seconds() / 86400) 
                           for t in observed_times])
        min_dt = dt_days.min()
        
        # Exponential decay
        rho_temporal = np.exp(-min_dt / temporal_range)
        
        # Spatial correlation
        rho_spatial = 0.0
        other_cols = [c for c in df_original.columns if c != col]
        
        for other_col in other_cols:
            if pd.notna(df_original.loc[miss_time, other_col]):
                corr_value = abs(corr_matrix.loc[col, other_col])
                rho_spatial = max(rho_spatial, corr_value)
        
        # Combine (using formula for independent contributions)
        rho_squared = (
            (1 - spatial_weight) * rho_temporal**2 +
            spatial_weight * rho_spatial**2
        )
        
        return min(rho_squared, 0.99)  # Cap at 0.99 to avoid numerical issues
    
    def _compute_average_kriging_reduction(
        self,
        df_original: pd.DataFrame,
        temporal_range: float,
        spatial_weight: float,
        corr_matrix: pd.DataFrame
    ) -> float:
        """
        Compute average ρ² across all missing values.
        """
        rho_squared_values = []
        
        for col in df_original.columns:
            missing_mask = df_original[col].isna()
            
            for miss_time in df_original.index[missing_mask]:
                rho_sq = self._compute_kriging_correlation(
                    df_original, col, miss_time,
                    temporal_range, spatial_weight, corr_matrix
                )
                rho_squared_values.append(rho_sq)
        
        if len(rho_squared_values) == 0:
            return 0.0
        
        return np.mean(rho_squared_values)
    
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
